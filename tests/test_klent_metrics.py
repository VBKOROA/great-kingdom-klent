from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
from great_kingdom_ai.klent.metrics import (
    END_REASON_KEYS,
    EpochMetrics,
    EpochMetricsAccumulator,
    SelfPlayMetrics,
    compute_self_play_metrics,
)

_torch_spec = importlib.util.find_spec("torch")
requires_torch = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None


def _episode(actions: list[int], end_reason: int) -> SimpleNamespace:
    transitions = tuple(SimpleNamespace(action=action) for action in actions)
    return SimpleNamespace(transitions=transitions, end_reason=end_reason)


def test_self_play_metrics_counts_passes_and_end_reasons() -> None:
    episodes = [
        _episode([0, 81, 1], end_reason=1),
        _episode([2, 3], end_reason=3),
        _episode([81], end_reason=99),
    ]

    metrics = compute_self_play_metrics(episodes)

    assert metrics.mean_game_length == pytest.approx(6 / 3)
    assert metrics.pass_count == 2
    assert metrics.pass_rate == pytest.approx(2 / 6)
    assert metrics.end_reason_counts == {
        "opponent_castle_destroyed": 1,
        "own_castle_destroyed": 0,
        "consecutive_passes": 1,
        "unknown_99": 1,
    }
    assert metrics.end_reason_rates["opponent_castle_destroyed"] == pytest.approx(1 / 3)
    assert metrics.end_reason_rates["own_castle_destroyed"] == 0.0
    assert metrics.end_reason_rates["consecutive_passes"] == pytest.approx(1 / 3)
    assert metrics.end_reason_rates["unknown_99"] == pytest.approx(1 / 3)
    assert sum(metrics.end_reason_rates.values()) == pytest.approx(1.0)


def test_self_play_metrics_emits_zero_count_known_reasons() -> None:
    metrics = compute_self_play_metrics([_episode([81], end_reason=3)])

    assert set(END_REASON_KEYS.values()) <= set(metrics.end_reason_counts)
    assert metrics.end_reason_counts["opponent_castle_destroyed"] == 0
    assert metrics.end_reason_counts["own_castle_destroyed"] == 0
    assert metrics.end_reason_counts["consecutive_passes"] == 1


def test_self_play_metrics_rejects_empty_episodes() -> None:
    with pytest.raises(ValueError, match="completed episode"):
        compute_self_play_metrics([])


def test_self_play_metrics_rejects_zero_transitions() -> None:
    with pytest.raises(ValueError, match="at least one transition"):
        compute_self_play_metrics([_episode([], end_reason=1)])


def test_self_play_metrics_to_dict_shape() -> None:
    metrics = compute_self_play_metrics([_episode([0, 1], end_reason=1)])

    payload = metrics.to_dict()

    assert set(payload) == {
        "mean_game_length",
        "pass_count",
        "pass_rate",
        "end_reason_counts",
        "end_reason_rates",
    }
    assert payload["pass_count"] == 0


def test_epoch_metrics_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="epoch_metrics\\[2\\].policy_kl"):
        EpochMetrics(
            epoch=2,
            policy_loss=1.0,
            q_loss=0.5,
            total_loss=1.5,
            target_policy_entropy=1.0,
            policy_kl=float("nan"),
        )


def test_self_play_metrics_rejects_non_finite_rate() -> None:
    with pytest.raises(ValueError, match="self_play_metrics.pass_rate"):
        SelfPlayMetrics(
            mean_game_length=2.0,
            pass_count=0,
            pass_rate=float("inf"),
            end_reason_counts={"opponent_castle_destroyed": 1},
            end_reason_rates={"opponent_castle_destroyed": 1.0},
        )


@requires_torch
def test_epoch_accumulator_matches_weighted_reference() -> None:
    batches = [
        {
            "policy": [0.5, 1.5, 2.0],
            "q": [0.1, 0.2, 0.3],
            "entropy": [0.4, 0.6, 0.8],
            "weights": [1.0, 2.0, 0.5],
        },
        {
            "policy": [0.25, 0.75],
            "q": [0.9, 1.1],
            "entropy": [0.2, 0.3],
            "weights": [3.0, 0.25],
        },
    ]
    accumulator = EpochMetricsAccumulator.create(torch, device="cpu")
    all_policy: list[float] = []
    all_q: list[float] = []
    all_entropy: list[float] = []
    all_weights: list[float] = []
    for batch in batches:
        accumulator.add(
            policy_loss=torch.tensor(batch["policy"]),
            q_loss=torch.tensor(batch["q"]),
            target_entropy=torch.tensor(batch["entropy"]),
            sample_weight=torch.tensor(batch["weights"]),
        )
        all_policy.extend(batch["policy"])
        all_q.extend(batch["q"])
        all_entropy.extend(batch["entropy"])
        all_weights.extend(batch["weights"])

    expected_total = sum(all_weights)
    expected_policy = (
        sum(value * weight for value, weight in zip(all_policy, all_weights, strict=True))
        / expected_total
    )
    expected_q = (
        sum(value * weight for value, weight in zip(all_q, all_weights, strict=True))
        / expected_total
    )
    expected_entropy = (
        sum(value * weight for value, weight in zip(all_entropy, all_weights, strict=True))
        / expected_total
    )

    metrics = accumulator.finalize(epoch=3)

    assert metrics.epoch == 3
    assert metrics.policy_loss == pytest.approx(expected_policy)
    assert metrics.q_loss == pytest.approx(expected_q)
    assert metrics.target_policy_entropy == pytest.approx(expected_entropy)
    assert metrics.total_loss == pytest.approx(expected_policy + expected_q)
    assert metrics.policy_kl == pytest.approx(expected_policy - expected_entropy)


@requires_torch
def test_epoch_metrics_to_dict_contains_all_fields() -> None:
    accumulator = EpochMetricsAccumulator.create(torch, device="cpu")
    accumulator.add(
        policy_loss=torch.tensor([1.0]),
        q_loss=torch.tensor([0.5]),
        target_entropy=torch.tensor([0.75]),
        sample_weight=torch.tensor([2.0]),
    )

    payload = accumulator.finalize(epoch=0).to_dict()

    assert payload == {
        "epoch": 0,
        "policy_loss": pytest.approx(1.0),
        "q_loss": pytest.approx(0.5),
        "total_loss": pytest.approx(1.5),
        "target_policy_entropy": pytest.approx(0.75),
        "policy_kl": pytest.approx(0.25),
    }


@requires_torch
def test_epoch_accumulator_rejects_zero_weight() -> None:
    accumulator = EpochMetricsAccumulator.create(torch, device="cpu")
    accumulator.add(
        policy_loss=torch.tensor([1.0]),
        q_loss=torch.tensor([0.5]),
        target_entropy=torch.tensor([0.25]),
        sample_weight=torch.tensor([0.0]),
    )

    with pytest.raises(ValueError, match="positive total sample weight"):
        accumulator.finalize(epoch=0)
