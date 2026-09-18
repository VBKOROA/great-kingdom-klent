"""Typed KLENT training diagnostics: epoch metrics and self-play statistics.

The values here are observability only. They never feed the optimization
objective, the stored targets, or the checkpoint/shard formats, so historical
logs can be interpreted without them and existing checkpoints stay resumable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from great_kingdom_ai.features import PASS_ACTION

if TYPE_CHECKING:
    from torch import Tensor

    from great_kingdom_ai.replay import TrajectoryEpisode

METRICS_SCHEMA_VERSION = 1

END_REASON_KEYS: dict[int, str] = {
    1: "opponent_castle_destroyed",
    2: "own_castle_destroyed",
    3: "consecutive_passes",
}


@dataclass(frozen=True)
class EpochMetrics:
    """Sample-weighted diagnostics measured before each optimizer update."""

    epoch: int
    policy_loss: float
    q_loss: float
    total_loss: float
    target_policy_entropy: float
    policy_kl: float

    def __post_init__(self) -> None:
        _require_finite(f"epoch_metrics[{self.epoch}].policy_loss", self.policy_loss)
        _require_finite(f"epoch_metrics[{self.epoch}].q_loss", self.q_loss)
        _require_finite(f"epoch_metrics[{self.epoch}].total_loss", self.total_loss)
        _require_finite(
            f"epoch_metrics[{self.epoch}].target_policy_entropy",
            self.target_policy_entropy,
        )
        _require_finite(f"epoch_metrics[{self.epoch}].policy_kl", self.policy_kl)

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "policy_loss": self.policy_loss,
            "q_loss": self.q_loss,
            "total_loss": self.total_loss,
            "target_policy_entropy": self.target_policy_entropy,
            "policy_kl": self.policy_kl,
        }


@dataclass
class EpochMetricsAccumulator:
    """Aggregate detached per-sample diagnostics on device.

    Each accumulated quantity is ``sum(sample_weight * value)`` and the
    finalization divides by ``sum(sample_weight)`` so unequal batch sizes and
    nonuniform sample weights contribute correctly.
    """

    torch: Any = field(repr=False)
    dtype: Any = field(repr=False)
    total_weight: Tensor
    weighted_policy_loss: Tensor
    weighted_q_loss: Tensor
    weighted_target_entropy: Tensor

    @classmethod
    def create(
        cls,
        torch: Any,
        *,
        device: Any,
        dtype: Any | None = None,
    ) -> EpochMetricsAccumulator:
        resolved_dtype = torch.float32 if dtype is None else dtype

        def zeros() -> Tensor:
            return cast("Tensor", torch.zeros((), dtype=resolved_dtype, device=device))

        return cls(
            torch=torch,
            dtype=resolved_dtype,
            total_weight=zeros(),
            weighted_policy_loss=zeros(),
            weighted_q_loss=zeros(),
            weighted_target_entropy=zeros(),
        )

    def add(
        self,
        *,
        policy_loss: Tensor,
        q_loss: Tensor,
        target_entropy: Tensor,
        sample_weight: Tensor,
    ) -> None:
        weights = sample_weight.to(dtype=self.dtype)
        self.total_weight = self.total_weight + weights.sum()
        self.weighted_policy_loss = self.weighted_policy_loss + _weighted_sum(
            policy_loss, weights, self.dtype
        )
        self.weighted_q_loss = self.weighted_q_loss + _weighted_sum(
            q_loss, weights, self.dtype
        )
        self.weighted_target_entropy = self.weighted_target_entropy + _weighted_sum(
            target_entropy, weights, self.dtype
        )

    def finalize(self, epoch: int) -> EpochMetrics:
        total_weight = _scalar(self.total_weight)
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError(
                "KLENT epoch metrics require a positive total sample weight, "
                f"got {total_weight!r}"
            )
        policy_loss = _scalar(self.weighted_policy_loss) / total_weight
        q_loss = _scalar(self.weighted_q_loss) / total_weight
        target_entropy = _scalar(self.weighted_target_entropy) / total_weight
        return EpochMetrics(
            epoch=epoch,
            policy_loss=policy_loss,
            q_loss=q_loss,
            total_loss=policy_loss + q_loss,
            target_policy_entropy=target_entropy,
            policy_kl=policy_loss - target_entropy,
        )


@dataclass(frozen=True)
class SelfPlayMetrics:
    """Collected-game statistics computed before augmentation."""

    mean_game_length: float
    pass_count: int
    pass_rate: float
    end_reason_counts: dict[str, int]
    end_reason_rates: dict[str, float]

    def __post_init__(self) -> None:
        _require_finite("self_play_metrics.mean_game_length", self.mean_game_length)
        _require_finite("self_play_metrics.pass_rate", self.pass_rate)
        for key, rate in self.end_reason_rates.items():
            _require_finite(f"self_play_metrics.end_reason_rates.{key}", rate)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_game_length": self.mean_game_length,
            "pass_count": self.pass_count,
            "pass_rate": self.pass_rate,
            "end_reason_counts": dict(self.end_reason_counts),
            "end_reason_rates": dict(self.end_reason_rates),
        }


def compute_self_play_metrics(
    episodes: Sequence[TrajectoryEpisode],
    *,
    pass_action: int = PASS_ACTION,
) -> SelfPlayMetrics:
    """Return aggregate statistics over completed, un-augmented episodes."""
    games = len(episodes)
    if games == 0:
        raise ValueError("KLENT self-play metrics require at least one completed episode")
    total_transitions = sum(len(episode.transitions) for episode in episodes)
    if total_transitions == 0:
        raise ValueError("KLENT self-play metrics require at least one transition")

    pass_count = 0
    known_counts = {code: 0 for code in END_REASON_KEYS}
    unknown_counts: dict[str, int] = {}
    for episode in episodes:
        for transition in episode.transitions:
            if int(transition.action) == pass_action:
                pass_count += 1
        code = int(episode.end_reason)
        if code in END_REASON_KEYS:
            known_counts[code] += 1
        else:
            key = f"unknown_{code}"
            unknown_counts[key] = unknown_counts.get(key, 0) + 1

    end_reason_counts = {END_REASON_KEYS[code]: known_counts[code] for code in END_REASON_KEYS}
    for key in sorted(unknown_counts):
        end_reason_counts[key] = unknown_counts[key]

    return SelfPlayMetrics(
        mean_game_length=total_transitions / games,
        pass_count=pass_count,
        pass_rate=pass_count / total_transitions,
        end_reason_counts=end_reason_counts,
        end_reason_rates={
            key: count / games for key, count in end_reason_counts.items()
        },
    )


def _weighted_sum(values: Tensor, weights: Tensor, dtype: Any) -> Tensor:
    return (values.to(dtype=dtype) * weights).sum()


def _scalar(tensor: Tensor) -> float:
    return float(tensor.detach().cpu())


def _require_finite(location: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"non-finite KLENT diagnostic at {location}: {value!r}")


__all__ = [
    "END_REASON_KEYS",
    "METRICS_SCHEMA_VERSION",
    "EpochMetrics",
    "EpochMetricsAccumulator",
    "SelfPlayMetrics",
    "compute_self_play_metrics",
]
