"""Synchronous KLENT collect/train iteration loop (paper Algorithm 1).

Each iteration freezes theta_k, collects a fresh whole-game buffer with
zero-search self-play, fits only on that buffer, writes the iteration
checkpoint, publishes actor/eval ONNX, and only then advances the latest
pointer. Resume recovers iterations whose training finished but whose ONNX
publication did not, instead of retraining or mixing partial state.
"""

from __future__ import annotations

import json
import random
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from great_kingdom_ai.augmentation import augment_training_arrays_randomly
from great_kingdom_ai.klent._io import copy_file_atomic
from great_kingdom_ai.klent._torch import _autocast_context, _cuda_amp_enabled, _import_torch
from great_kingdom_ai.klent.batch import TrainingArrays, TrainingBatch, arrays_to_batch
from great_kingdom_ai.klent.checkpoint import (
    KlentCheckpointMetadata,
    KlentTrainState,
    load_klent_checkpoint,
    read_klent_checkpoint_metadata,
    save_klent_checkpoint,
    warm_start_klent_model,
)
from great_kingdom_ai.klent.dataset import KlentReplayDataset
from great_kingdom_ai.klent.loss import KlentLossBreakdown, compute_klent_losses
from great_kingdom_ai.klent.metrics import (
    METRICS_SCHEMA_VERSION,
    EpochMetrics,
    EpochMetricsAccumulator,
    SelfPlayMetrics,
    compute_self_play_metrics,
)
from great_kingdom_ai.klent.publish import (
    load_published_klent_onnx,
    publish_klent_onnx_artifacts,
)
from great_kingdom_ai.klent.self_play import KlentSelfPlayConfig, play_klent_game
from great_kingdom_ai.klent.shards import (
    KlentShardMetadata,
    save_klent_shard,
    shard_metadata_path,
)
from great_kingdom_ai.klent.types import KlentConfig, KlentPolicyValueModel
from great_kingdom_ai.replay.trajectory import TrajectoryReplayStore

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

    from great_kingdom_ai.replay import TrajectoryEpisode

_ACTOR_OVERRIDE_TOLERANCE = 1e-2


@dataclass(frozen=True)
class KlentTrainConfig:
    work_dir: Path = Path("data/klent")
    klent: KlentConfig = KlentConfig()
    model_preset: str = "strong_attn_klent"
    device: str = "cpu"
    seed: int = 0
    min_transitions: int = 4096
    max_games_per_iteration: int = 256
    fit_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    gradient_clip_norm: float | None = None
    max_turns: int = 200
    symmetry_augmentation: bool = True
    amp: bool = False
    keep_shards: bool = True
    export_onnx: bool = True
    onnx_device: str = "cpu"
    onnx_precision: str = "fp32"
    check_onnx_parity: bool = True
    use_rust_actor: bool = False
    actor_onnx_path: Path | None = None
    rust_self_play_batch_size: int = 64
    warm_start_checkpoint: Path | None = None
    override_learning_rate: bool = False

    def __post_init__(self) -> None:
        if self.min_transitions <= 0:
            raise ValueError("min_transitions must be positive")
        if self.max_games_per_iteration <= 0:
            raise ValueError("max_games_per_iteration must be positive")
        if self.fit_epochs <= 0:
            raise ValueError("fit_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.gradient_clip_norm is not None and (
            not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")
        if self.onnx_device not in {"cpu", "cuda"}:
            raise ValueError("onnx_device must be one of: cpu, cuda")
        if self.onnx_precision not in {"fp32", "fp16"}:
            raise ValueError("onnx_precision must be one of: fp32, fp16")
        if self.rust_self_play_batch_size <= 0:
            raise ValueError("rust_self_play_batch_size must be positive")
        if self.export_onnx and self.onnx_precision == "fp16" and self.check_onnx_parity:
            raise ValueError(
                "CPU parity checks only support fp32 ONNX; "
                "disable check_onnx_parity for fp16 publication"
            )


@dataclass(frozen=True)
class KlentIterationSummary:
    iteration: int
    games: int
    transitions: int
    shard_path: Path
    checkpoint_path: Path
    epoch_losses: list[float]
    epoch_metrics: list[EpochMetrics]
    self_play_metrics: SelfPlayMetrics
    onnx_version_dir: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "games": self.games,
            "transitions": self.transitions,
            "shard_path": str(self.shard_path),
            "checkpoint_path": str(self.checkpoint_path),
            "epoch_losses": self.epoch_losses,
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "epoch_metrics": [metrics.to_dict() for metrics in self.epoch_metrics],
            "self_play_metrics": self.self_play_metrics.to_dict(),
            "onnx_version_dir": (
                None if self.onnx_version_dir is None else str(self.onnx_version_dir)
            ),
        }


@dataclass(frozen=True)
class KlentFitResult:
    """Detailed training result while ``fit_klent_model`` keeps its tuple API."""

    epoch_losses: list[float]
    epoch_metrics: list[EpochMetrics]
    steps: int


def run_klent_training(
    config: KlentTrainConfig,
    *,
    iterations: int,
    resume: bool = True,
) -> list[KlentIterationSummary]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    return list(iter_klent_training(config, iterations=iterations, resume=resume))


def iter_klent_training(
    config: KlentTrainConfig,
    *,
    iterations: int | None = None,
    resume: bool = True,
) -> Iterator[KlentIterationSummary]:
    """Yield completed iterations; None runs until interrupted without retaining history."""
    if iterations is not None and iterations <= 0:
        raise ValueError("iterations must be positive")
    config.work_dir.mkdir(parents=True, exist_ok=True)
    _iteration_shard_path(config.work_dir, 0).parent.mkdir(parents=True, exist_ok=True)
    _iteration_checkpoint_path(config.work_dir, 0).parent.mkdir(parents=True, exist_ok=True)

    state = _initial_state(config, resume=resume)
    while iterations is None or state.iteration < iterations:
        state, summary = run_klent_iteration(state, config)
        yield summary


def run_klent_iteration(
    state: KlentTrainState,
    config: KlentTrainConfig,
) -> tuple[KlentTrainState, KlentIterationSummary]:
    torch = _import_torch()
    iteration = state.iteration
    model = state.model
    model.eval()
    if config.use_rust_actor:
        episodes = _collect_with_rust_actor(state, config, iteration)
    else:
        episodes = _collect_with_python_actor(model, config, iteration)
    transition_count = sum(len(episode.transitions) for episode in episodes)
    if not episodes:
        raise RuntimeError("KLENT collection produced no episodes")
    self_play_metrics = compute_self_play_metrics(episodes)

    store = TrajectoryReplayStore.from_episodes(max(transition_count, 1), episodes)
    metadata = KlentShardMetadata.from_config(
        config.klent,
        iteration=iteration,
        model_version=iteration,
        transitions=len(store),
        games=len(episodes),
    )
    shard_path = _iteration_shard_path(config.work_dir, iteration)
    save_klent_shard(store, shard_path, metadata)

    dataset = KlentReplayDataset(store, config=config.klent, metadata=metadata)
    model.train()
    fit_result = fit_klent_model_detailed(
        model,
        dataset,
        config,
        state.optimizer,
        start_steps=state.total_steps,
        iteration=iteration,
        scaler=state.scaler,
    )
    model.eval()
    total_steps = fit_result.steps

    next_iteration = iteration + 1
    next_state = KlentTrainState(
        model=model,
        optimizer=state.optimizer,
        iteration=next_iteration,
        total_steps=total_steps,
        klent_config=config.klent,
        model_preset=config.model_preset,
        last_shard=str(shard_path),
        scaler=state.scaler,
        run_id=state.run_id,
    )
    checkpoint_path = _iteration_checkpoint_path(config.work_dir, next_iteration)
    save_klent_checkpoint(next_state, checkpoint_path)
    onnx_version_dir: Path | None = None
    if config.export_onnx:
        manifest = publish_klent_onnx_artifacts(
            checkpoint_path,
            config.work_dir,
            model_version=next_iteration,
            iteration=iteration,
            klent_config=config.klent,
            model_preset=config.model_preset,
            device=config.onnx_device,
            precision=config.onnx_precision,
            check_parity=config.check_onnx_parity,
            overwrite=True,
            run_id=next_state.run_id,
        )
        onnx_version_dir = _onnx_version_dir(config.work_dir, manifest.model_version)
    _publish_latest_checkpoint(config.work_dir, next_iteration)
    if not config.keep_shards:
        metadata_path = shard_metadata_path(shard_path)
        if metadata_path.exists():
            metadata_path.unlink()
        if shard_path.exists():
            shard_path.unlink()

    del torch
    return next_state, KlentIterationSummary(
        iteration=iteration,
        games=len(episodes),
        transitions=len(store),
        shard_path=shard_path,
        checkpoint_path=checkpoint_path,
        epoch_losses=fit_result.epoch_losses,
        epoch_metrics=fit_result.epoch_metrics,
        self_play_metrics=self_play_metrics,
        onnx_version_dir=onnx_version_dir,
    )


def fit_klent_model(
    model: nn.Module,
    dataset: KlentReplayDataset,
    config: KlentTrainConfig,
    optimizer: Optimizer,
    *,
    start_steps: int,
    iteration: int,
    scaler: Any | None = None,
) -> tuple[list[float], int]:
    """Fit and return the historical ``(epoch_losses, steps)`` tuple."""
    result = fit_klent_model_detailed(
        model,
        dataset,
        config,
        optimizer,
        start_steps=start_steps,
        iteration=iteration,
        scaler=scaler,
    )
    return result.epoch_losses, result.steps


def fit_klent_model_detailed(
    model: nn.Module,
    dataset: KlentReplayDataset,
    config: KlentTrainConfig,
    optimizer: Optimizer,
    *,
    start_steps: int,
    iteration: int,
    scaler: Any | None = None,
) -> KlentFitResult:
    """Fit for ``fit_epochs`` shuffled passes over the frozen iteration buffer.

    Diagnostics accumulate detached per-sample scalars on device and are read
    back once per epoch. ``epoch_losses`` keeps the historical unweighted mean
    of batch total-loss means; ``epoch_metrics`` aggregates the same per-sample
    losses with ``sum(sample_weight * value) / sum(sample_weight)``.
    """
    torch = _import_torch()
    amp_enabled = _cuda_amp_enabled(torch, config.device, enabled=config.amp)
    scaler = scaler if amp_enabled else None
    permutation_rng = np.random.default_rng(config.seed + iteration)
    augment_rng = random.Random(config.seed + iteration)
    steps = start_steps
    epoch_losses: list[float] = []
    epoch_metrics: list[EpochMetrics] = []
    for epoch in range(config.fit_epochs):
        order = permutation_rng.permutation(len(dataset))
        batch_total = 0.0
        batch_count = 0
        accumulator: EpochMetricsAccumulator | None = None
        for start in range(0, len(order), config.batch_size):
            indexes = np.asarray(order[start : start + config.batch_size], dtype=np.int64)
            batch = _klent_batch_from_indexes(dataset, indexes, config, augment_rng)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(torch, enabled=amp_enabled):
                losses = compute_klent_losses(model, batch, config.klent)
            if accumulator is None:
                accumulator = EpochMetricsAccumulator.create(
                    torch,
                    device=batch.sample_weight.device,
                )
            accumulator.add(
                policy_loss=losses.per_sample_policy_loss.detach(),
                q_loss=losses.per_sample_q_error.detach(),
                target_entropy=losses.per_sample_target_entropy,
                sample_weight=batch.sample_weight.detach(),
            )
            if scaler is not None:
                scaler.scale(losses.total).backward()
                if config.gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.total.backward()  # type: ignore[no-untyped-call]
                if config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
                optimizer.step()
            steps += 1
            batch_total += float(losses.total.detach().cpu())
            batch_count += 1
        if accumulator is None:
            raise ValueError("KLENT fitting produced no batches for epoch metrics")
        epoch_losses.append(batch_total / max(batch_count, 1))
        epoch_metrics.append(accumulator.finalize(epoch))
    return KlentFitResult(
        epoch_losses=epoch_losses,
        epoch_metrics=epoch_metrics,
        steps=steps,
    )


def compute_iteration_loss(
    model: nn.Module,
    dataset: KlentReplayDataset,
    config: KlentTrainConfig,
) -> KlentLossBreakdown:
    """Compute the loss over the whole buffer in one batch for diagnostics."""
    indexes = np.arange(len(dataset), dtype=np.int64)
    batch = _klent_batch_from_indexes(dataset, indexes, config, random.Random(0))
    return compute_klent_losses(model, batch, config.klent)


def _collect_with_python_actor(
    model: KlentPolicyValueModel,
    config: KlentTrainConfig,
    iteration: int,
) -> list[TrajectoryEpisode]:
    episodes: list[TrajectoryEpisode] = []
    transition_count = 0
    seed_base = config.seed + iteration * 1_000_003
    while transition_count < config.min_transitions:
        if len(episodes) >= config.max_games_per_iteration:
            raise RuntimeError(
                f"collected {transition_count} transitions in {len(episodes)} games "
                f"before reaching min_transitions={config.min_transitions}; "
                "increase max_games_per_iteration or lower min_transitions"
            )
        _log, episode = play_klent_game(
            model,
            seed=seed_base + len(episodes),
            config=config.klent,
            self_play=KlentSelfPlayConfig(
                max_turns=config.max_turns,
                episode_id=len(episodes),
                model_version=iteration,
                created_iteration=iteration,
            ),
        )
        episodes.append(episode)
        transition_count += len(episode.transitions)
    return episodes


def _collect_with_rust_actor(
    state: KlentTrainState,
    config: KlentTrainConfig,
    iteration: int,
) -> list[TrajectoryEpisode]:
    from great_kingdom_ai.klent.export import export_klent_checkpoint_to_onnx
    from great_kingdom_ai.klent.rust_actor import (
        RustKlentActorConfig,
        play_rust_klent_zero_search,
    )

    actor_path = _actor_onnx_for_iteration(
        state,
        config,
        iteration,
        export_klent_checkpoint_to_onnx,
    )

    episodes: list[TrajectoryEpisode] = []
    transition_count = 0
    seed_base = config.seed + iteration * 1_000_003
    game_offset = 0
    while transition_count < config.min_transitions:
        if game_offset >= config.max_games_per_iteration:
            raise RuntimeError(
                f"collected {transition_count} transitions in {game_offset} games "
                f"before reaching min_transitions={config.min_transitions}; "
                "increase max_games_per_iteration or lower min_transitions"
            )
        batch_size = min(
            config.rust_self_play_batch_size,
            config.max_games_per_iteration - game_offset,
        )
        summary = play_rust_klent_zero_search(
            RustKlentActorConfig(
                actor_onnx_path=actor_path,
                output_dir=config.work_dir,
                games=batch_size,
                seed_start=seed_base + game_offset,
                alpha=config.klent.alpha,
                beta=config.klent.beta,
                lambda_param=config.klent.lambda_param,
                gamma=config.klent.gamma,
                max_turns=config.max_turns,
                onnx_device=config.onnx_device,
                onnx_max_batch_size=batch_size,
                rust_self_play_batch_size=batch_size,
                model_version=iteration,
                created_iteration=iteration,
                episode_id_offset=game_offset,
            )
        )
        episodes.extend(summary.trajectory_episodes)
        transition_count += summary.transitions
        game_offset += batch_size
    return episodes


def _actor_onnx_for_iteration(
    state: KlentTrainState,
    config: KlentTrainConfig,
    iteration: int,
    export_checkpoint_to_onnx: Callable[..., Any],
) -> Path:
    """Return the actor model that matches the frozen iteration model.

    ``actor_onnx_path`` is only an initial (iteration 0) override, and it is
    used only after its outputs are verified against the learner checkpoint so
    the shard records and the collected behavior cannot diverge. Later
    iterations always export an actor from their own frozen checkpoint.
    """
    if config.actor_onnx_path is not None and iteration == 0:
        provided = config.actor_onnx_path
        if not provided.exists():
            raise FileNotFoundError(f"actor ONNX model is missing: {provided}")
        source_checkpoint = _actor_source_checkpoint_path(config.work_dir, iteration)
        save_klent_checkpoint(state, source_checkpoint)
        _verify_actor_override(source_checkpoint, provided)
        return provided

    source_checkpoint = _actor_source_checkpoint_path(config.work_dir, iteration)
    save_klent_checkpoint(state, source_checkpoint)
    actor_path = config.work_dir / "onnx" / f"actor-source-{iteration:04d}.onnx"
    export_checkpoint_to_onnx(
        source_checkpoint,
        actor_path,
        kind="actor",
        device=config.onnx_device,
        precision=config.onnx_precision,
    )
    return actor_path


def _verify_actor_override(checkpoint_path: Path, actor_onnx_path: Path) -> None:
    from great_kingdom_ai.klent.export import compare_klent_checkpoint_to_onnx

    try:
        summary = compare_klent_checkpoint_to_onnx(
            checkpoint_path,
            actor_onnx_path,
            kind="actor",
            tolerance=_ACTOR_OVERRIDE_TOLERANCE,
        )
    except ValueError as error:
        raise RuntimeError(
            f"actor_onnx_path {actor_onnx_path} cannot be compared with the "
            f"iteration 0 learner checkpoint: {error}"
        ) from error
    if summary.passed:
        return
    raise RuntimeError(
        f"actor_onnx_path {actor_onnx_path} does not match the iteration 0 "
        f"learner checkpoint (policy diff {summary.max_policy_abs_diff:.3g}, "
        f"value diff {summary.max_value_abs_diff:.3g}); start the learner from "
        "the matching warm_start_checkpoint or drop actor_onnx_path"
    )


def _actor_source_checkpoint_path(work_dir: Path, iteration: int) -> Path:
    return work_dir / "checkpoints" / f"actor-source-{iteration:04d}.pt"


def _klent_batch_from_indexes(
    dataset: KlentReplayDataset,
    indexes: np.ndarray,
    config: KlentTrainConfig,
    augment_rng: random.Random,
) -> TrainingBatch:
    batch = dataset.arrays_for_indexes(indexes)
    arrays = TrainingArrays(
        features=batch.features,
        policies=batch.policies,
        values=batch.values,
        sample_weights=batch.sample_weights,
        legal_masks=batch.legal_masks,
        indexes=batch.indexes,
        actions=batch.actions,
    )
    if config.symmetry_augmentation:
        features, policies, legal_masks, _terminal, actions = augment_training_arrays_randomly(
            arrays.features,
            arrays.policies,
            arrays.legal_masks,
            augment_rng,
            actions=arrays.actions,
        )
        arrays = replace(
            arrays,
            features=features,
            policies=policies,
            legal_masks=legal_masks,
            actions=actions,
        )
    return arrays_to_batch(
        arrays,
        device=config.device,
        pin_memory=str(config.device).startswith("cuda"),
    )


def _initial_state(config: KlentTrainConfig, *, resume: bool) -> KlentTrainState:
    torch = _import_torch()
    from great_kingdom_ai.klent.optim import OptimizerConfig, create_optimizer
    from great_kingdom_ai.model import create_model

    run_id, allow_legacy = _resolve_run_id(config, resume=resume)
    if resume:
        recovered = _resume_checkpoint(config, run_id, allow_legacy=allow_legacy)
        if recovered is not None:
            return recovered

    if config.warm_start_checkpoint is not None:
        model = warm_start_klent_model(
            config.warm_start_checkpoint,
            model_preset=config.model_preset,
            device=config.device,
        )
        if not model.has_action_value_head:
            raise ValueError("KLENT warm start requires action_value_head=True")
    else:
        model = create_model(config.model_preset)
        if not model.has_action_value_head:
            raise ValueError(
                f"KLENT model preset {config.model_preset!r} must enable action_value_head"
            )
        model = model.to(config.device)

    optimizer = create_optimizer(
        torch,
        model,
        OptimizerConfig(
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
    )
    scaler: Any | None = None
    if _cuda_amp_enabled(torch, config.device, enabled=config.amp):
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    return KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=0,
        total_steps=0,
        klent_config=config.klent,
        model_preset=config.model_preset,
        scaler=scaler,
        run_id=run_id,
    )


def _load_state(path: Path, config: KlentTrainConfig) -> KlentTrainState:
    state = load_klent_checkpoint(
        path,
        device=config.device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        amp=config.amp,
    )
    if state.klent_config != config.klent:
        raise ValueError("resume KLENT config does not match the checkpoint")
    if config.override_learning_rate:
        for group in state.optimizer.param_groups:
            group["lr"] = config.learning_rate
    return state


@dataclass(frozen=True)
class _ResumeCandidate:
    path: Path
    iteration: int
    total_steps: int
    is_iteration_checkpoint: bool


def _resume_checkpoint(
    config: KlentTrainConfig,
    run_id: str,
    *,
    allow_legacy: bool = False,
) -> KlentTrainState | None:
    """Resume from the newest usable checkpoint of the active run.

    Recovery candidates include all healthy iteration checkpoints and
    latest.pt belonging to the active run. When resuming a directory whose
    checkpoints predate run identifiers (empty run_id), the newest state is
    adopted and migrated to the active run_id only if allow_legacy is True.
    Missing or corrupt files are repaired so both the canonical iteration
    checkpoint and latest.pt are in sync.
    """
    candidates: list[_ResumeCandidate] = []
    legacy_candidates: list[_ResumeCandidate] = []
    for iteration, path in _iteration_checkpoints(config.work_dir):
        meta = _readable_checkpoint_metadata(path)
        if meta is None:
            continue
        if meta.run_id != run_id and (not allow_legacy or meta.run_id != ""):
            continue
        if meta.iteration != iteration:
            raise ValueError(
                f"iteration checkpoint {path} stores iteration {meta.iteration} "
                f"but is named for {iteration}"
            )
        candidate = _ResumeCandidate(
            path=path,
            iteration=meta.iteration,
            total_steps=meta.total_steps,
            is_iteration_checkpoint=True,
        )
        if meta.run_id == run_id:
            candidates.append(candidate)
        elif allow_legacy and meta.run_id == "":
            legacy_candidates.append(candidate)

    latest_path = _latest_checkpoint_path(config.work_dir)
    if latest_path.exists():
        meta = _readable_checkpoint_metadata(latest_path)
        if meta is not None:
            candidate = _ResumeCandidate(
                path=latest_path,
                iteration=meta.iteration,
                total_steps=meta.total_steps,
                is_iteration_checkpoint=False,
            )
            if meta.run_id == run_id:
                candidates.append(candidate)
            elif allow_legacy and meta.run_id == "":
                legacy_candidates.append(candidate)

    is_migrating_legacy = False
    if candidates:
        chosen_pool = candidates
    elif allow_legacy and legacy_candidates:
        chosen_pool = legacy_candidates
        is_migrating_legacy = True
    else:
        return None

    best = max(
        chosen_pool,
        key=lambda c: (c.iteration, c.total_steps, 1 if c.is_iteration_checkpoint else 0),
    )
    state = _load_state(best.path, config)
    if is_migrating_legacy:
        state = replace(state, run_id=run_id)

    canonical_iteration_path = _iteration_checkpoint_path(config.work_dir, state.iteration)
    if is_migrating_legacy:
        save_klent_checkpoint(state, canonical_iteration_path)
    elif (
        not canonical_iteration_path.exists()
        or _readable_checkpoint_run_id(canonical_iteration_path) != run_id
    ):
        copy_file_atomic(best.path, canonical_iteration_path)

    _publish_latest_checkpoint(config.work_dir, state.iteration)
    _ensure_published(config, state, source_path=canonical_iteration_path)

    if allow_legacy:
        _migrate_legacy_checkpoints(config.work_dir, run_id, exclude={canonical_iteration_path})
        _write_run_marker(_run_marker_path(config.work_dir), run_id, allow_legacy=False)

    return state


_resume_from_iteration_checkpoint = _resume_checkpoint


def _migrate_legacy_checkpoints(
    work_dir: Path,
    run_id: str,
    *,
    exclude: set[Path],
) -> None:
    for _iteration, path in _iteration_checkpoints(work_dir):
        if path in exclude or not path.exists():
            continue
        meta = _readable_checkpoint_metadata(path)
        if meta is not None and meta.run_id == "":
            _stamp_checkpoint_run_id(path, run_id)


def _stamp_checkpoint_run_id(path: Path, run_id: str) -> None:
    torch = _import_torch()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["run_id"] = run_id
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_published(
    config: KlentTrainConfig,
    state: KlentTrainState,
    *,
    source_path: Path | None = None,
) -> None:
    if not config.export_onnx:
        return
    try:
        pointer, manifest = load_published_klent_onnx(config.work_dir)
    except (ValueError, OSError, json.JSONDecodeError):
        pointer = None
        manifest = None
    if (
        pointer is not None
        and manifest is not None
        and pointer.run_id == state.run_id
        and manifest.run_id == state.run_id
        and pointer.model_version >= state.iteration
    ):
        return
    checkpoint_path = (
        source_path
        if source_path is not None
        else _iteration_checkpoint_path(config.work_dir, state.iteration)
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"KLENT checkpoint for publication is missing: {checkpoint_path}")
    publish_klent_onnx_artifacts(
        checkpoint_path,
        config.work_dir,
        model_version=state.iteration,
        iteration=state.iteration - 1,
        klent_config=config.klent,
        model_preset=config.model_preset,
        device=config.onnx_device,
        precision=config.onnx_precision,
        check_parity=config.check_onnx_parity,
        overwrite=True,
        run_id=state.run_id,
    )


def _publish_latest_checkpoint(work_dir: Path, iteration: int) -> None:
    checkpoint_path = _iteration_checkpoint_path(work_dir, iteration)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"iteration checkpoint is missing: {checkpoint_path}")
    copy_file_atomic(checkpoint_path, _latest_checkpoint_path(work_dir))


def _resolve_run_id(config: KlentTrainConfig, *, resume: bool) -> tuple[str, bool]:
    """Return the active run id and whether legacy checkpoints may be adopted.

    Iteration checkpoints embed the run id so a ``--no-resume`` restart cannot
    later resume into the previous run's checkpoints.
    """
    marker = _run_marker_path(config.work_dir)
    if resume and marker.exists():
        marker_run_id, allow_legacy = _read_run_marker(marker)
        if marker_run_id:
            return marker_run_id, allow_legacy
    if resume:
        has_checkpoints, newest_run_id = _inspect_existing_checkpoints(config.work_dir)
        if has_checkpoints and newest_run_id:
            run_id = newest_run_id
            allow_legacy = False
        elif has_checkpoints and not newest_run_id:
            run_id = uuid.uuid4().hex
            allow_legacy = True
        else:
            run_id = uuid.uuid4().hex
            allow_legacy = False
    else:
        run_id = uuid.uuid4().hex
        allow_legacy = False
    _write_run_marker(marker, run_id, allow_legacy=allow_legacy)
    return run_id, allow_legacy


def _inspect_existing_checkpoints(work_dir: Path) -> tuple[bool, str]:
    """Return whether readable checkpoints exist and the newest checkpoint's run id."""
    candidates: list[tuple[int, int, str]] = []
    for _iteration, path in _iteration_checkpoints(work_dir):
        meta = _readable_checkpoint_metadata(path)
        if meta is not None:
            candidates.append((meta.iteration, meta.total_steps, meta.run_id))
    latest = _latest_checkpoint_path(work_dir)
    if latest.exists():
        meta = _readable_checkpoint_metadata(latest)
        if meta is not None:
            candidates.append((meta.iteration, meta.total_steps, meta.run_id))
    if not candidates:
        return False, ""
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return True, candidates[0][2]


def _newest_known_run_id(work_dir: Path) -> str:
    """Adopt the run id of an existing directory that predates run markers."""
    has_checkpoints, run_id = _inspect_existing_checkpoints(work_dir)
    return run_id if has_checkpoints else ""


def _readable_checkpoint_metadata(path: Path) -> KlentCheckpointMetadata | None:
    try:
        return read_klent_checkpoint_metadata(path)
    except Exception:
        return None


def _readable_checkpoint_run_id(path: Path) -> str | None:
    meta = _readable_checkpoint_metadata(path)
    return None if meta is None else meta.run_id


def _run_marker_path(work_dir: Path) -> Path:
    return work_dir / "checkpoints" / "run.json"


def _read_run_marker(path: Path) -> tuple[str, bool]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "run_id" not in data:
        raise ValueError(f"KLENT run marker is malformed: {path}")
    return str(data["run_id"]), bool(data.get("allow_legacy", False))


def _read_run_id(path: Path) -> str:
    run_id, _ = _read_run_marker(path)
    return run_id


def _write_run_marker(path: Path, run_id: str, *, allow_legacy: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload: dict[str, Any] = {"run_id": run_id}
    if allow_legacy:
        payload["allow_legacy"] = True
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_run_id(path: Path, run_id: str) -> None:
    _write_run_marker(path, run_id, allow_legacy=False)


def _iteration_checkpoints(work_dir: Path) -> list[tuple[int, Path]]:
    directory = work_dir / "checkpoints"
    if not directory.exists():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for path in directory.glob("iteration-*.pt"):
        suffix = path.stem.removeprefix("iteration-")
        if not suffix.isdigit():
            continue
        checkpoints.append((int(suffix), path))
    checkpoints.sort()
    return checkpoints


def _iteration_shard_path(work_dir: Path, iteration: int) -> Path:
    return work_dir / "iterations" / f"iteration-{iteration:04d}.npz"


def _iteration_checkpoint_path(work_dir: Path, iteration: int) -> Path:
    return work_dir / "checkpoints" / f"iteration-{iteration:04d}.pt"


def _latest_checkpoint_path(work_dir: Path) -> Path:
    return work_dir / "checkpoints" / "latest.pt"


def _onnx_version_dir(work_dir: Path, model_version: int) -> Path:
    return work_dir / "onnx" / f"version-{model_version:05d}"


__all__ = [
    "KlentFitResult",
    "KlentIterationSummary",
    "KlentTrainConfig",
    "compute_iteration_loss",
    "fit_klent_model",
    "fit_klent_model_detailed",
    "iter_klent_training",
    "run_klent_iteration",
    "run_klent_training",
]
