from __future__ import annotations

import importlib
import importlib.util
import json
import unittest.mock
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

_torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    _torch_spec is None,
    reason="torch is not installed",
)
torch = importlib.import_module("torch") if _torch_spec is not None else None

import great_kingdom_ai.klent.trainer as trainer_module  # noqa: E402
from great_kingdom_ai.features import ACTION_SPACE, BOARD_SIZE, FEATURE_CHANNELS  # noqa: E402
from great_kingdom_ai.klent.checkpoint import (  # noqa: E402
    KlentTrainState,
    load_klent_checkpoint,
    save_klent_checkpoint,
    warm_start_klent_model,
)
from great_kingdom_ai.klent.optim import OptimizerConfig, create_optimizer  # noqa: E402
from great_kingdom_ai.klent.publish import (  # noqa: E402
    load_klent_onnx_manifest,
    load_klent_onnx_pointer,
)
from great_kingdom_ai.klent.shards import load_klent_shard  # noqa: E402
from great_kingdom_ai.klent.trainer import (  # noqa: E402
    KlentTrainConfig,
    _actor_onnx_for_iteration,
    _collect_with_python_actor,
    fit_klent_model,
    run_klent_training,
)
from great_kingdom_ai.klent.types import KlentConfig  # noqa: E402
from great_kingdom_ai.model import create_model  # noqa: E402
from great_kingdom_ai.replay.dataset import TrajectoryArrayBatch  # noqa: E402

_rust_core_available = importlib.util.find_spec("great_kingdom_core") is not None
requires_core = pytest.mark.skipif(
    not _rust_core_available,
    reason="great_kingdom_core extension is not installed",
)
_onnx_ready = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)
requires_onnx = pytest.mark.skipif(
    not _onnx_ready,
    reason="onnx and onnxruntime are required for ONNX publication",
)


def make_config(tmp_path: Path, **overrides: object) -> KlentTrainConfig:
    defaults: dict[str, object] = {
        "work_dir": tmp_path / "klent",
        "model_preset": "small_klent",
        "device": "cpu",
        "seed": 0,
        "min_transitions": 1,
        "max_games_per_iteration": 64,
        "fit_epochs": 2,
        "batch_size": 4,
        "max_turns": 200,
        "symmetry_augmentation": False,
        "learning_rate": 1e-2,
        "export_onnx": False,
    }
    return KlentTrainConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


@requires_core
def test_run_klent_training_publishes_iteration_checkpoints_and_shards(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    summaries = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in summaries] == [0, 1]
    assert all(summary.transitions >= config.min_transitions for summary in summaries)
    assert all(summary.epoch_losses for summary in summaries)
    latest = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest.iteration == 2

    for iteration in (0, 1):
        shard_path = config.work_dir / "iterations" / f"iteration-{iteration:04d}.npz"
        checkpoint_path = (
            config.work_dir / "checkpoints" / f"iteration-{iteration + 1:04d}.pt"
        )
        assert shard_path.exists()
        assert checkpoint_path.exists()
        store, metadata = load_klent_shard(shard_path)
        assert metadata.iteration == iteration
        assert metadata.model_version == iteration
        assert metadata.transitions == len(store)
        assert store.model_versions.tolist() == [iteration] * len(store)
        assert store.lambda_returns_present is not None
        assert bool(store.lambda_returns_present.all())


@requires_core
def test_iteration_summary_reports_training_and_self_play_metrics(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        fit_epochs=2,
        min_transitions=8,
    )

    summary = run_klent_training(config, iterations=1)[0]
    payload = summary.to_dict()

    assert payload["metrics_schema_version"] == 1
    epoch_metrics = payload["epoch_metrics"]
    assert isinstance(epoch_metrics, list)
    assert len(epoch_metrics) == config.fit_epochs
    assert [entry["epoch"] for entry in epoch_metrics] == [0, 1]
    for entry in epoch_metrics:
        for key in (
            "policy_loss",
            "q_loss",
            "total_loss",
            "target_policy_entropy",
            "policy_kl",
        ):
            assert np.isfinite(entry[key])
        assert entry["total_loss"] == pytest.approx(entry["policy_loss"] + entry["q_loss"])
        assert entry["policy_kl"] == pytest.approx(
            entry["policy_loss"] - entry["target_policy_entropy"]
        )

    self_play = payload["self_play_metrics"]
    assert self_play["mean_game_length"] == pytest.approx(
        summary.transitions / summary.games
    )
    assert self_play["pass_rate"] == pytest.approx(
        self_play["pass_count"] / summary.transitions
    )
    assert sum(self_play["end_reason_counts"].values()) == summary.games
    assert sum(self_play["end_reason_rates"].values()) == pytest.approx(1.0)


@requires_core
def test_run_klent_training_resumes_from_latest_checkpoint(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=1)

    resumed = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in resumed] == [1]
    latest = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest.iteration == 2
    assert latest.total_steps > 0
    assert latest.last_shard is not None


@requires_core
def test_fresh_run_does_not_adopt_previous_run_checkpoints(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=2)
    latest_path = config.work_dir / "checkpoints" / "latest.pt"

    run_klent_training(config, iterations=1, resume=False)
    fresh = load_klent_checkpoint(latest_path)
    assert fresh.iteration == 1

    resumed = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in resumed] == [1]
    trained = load_klent_checkpoint(latest_path)
    assert trained.iteration == 2
    assert trained.total_steps > fresh.total_steps


@requires_core
def test_resume_skips_corrupt_iteration_checkpoint(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=1)
    corrupt = config.work_dir / "checkpoints" / "iteration-0002.pt"
    corrupt.write_bytes(b"truncated checkpoint")

    resumed = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in resumed] == [1]
    repaired = load_klent_checkpoint(corrupt)
    assert repaired.iteration == 2


@requires_core
def test_resume_falls_back_to_latest_checkpoint(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=1)
    (config.work_dir / "checkpoints" / "iteration-0001.pt").unlink()

    resumed = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in resumed] == [1]
    latest = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest.iteration == 2


@requires_core
@requires_onnx
def test_resume_republishes_from_latest_checkpoint(tmp_path: Path) -> None:
    config = make_config(tmp_path, export_onnx=True, onnx_precision="fp32")
    run_klent_training(config, iterations=1)
    (config.work_dir / "checkpoints" / "iteration-0001.pt").unlink()
    (config.work_dir / "onnx" / "current.json").unlink()

    resumed = run_klent_training(config, iterations=1)

    assert resumed == []
    pointer = load_klent_onnx_pointer(config.work_dir)
    assert pointer.model_version == 1


@requires_core
def test_run_klent_training_resume_rejects_config_mismatch(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=1)

    mismatched = make_config(tmp_path, klent=KlentConfig(alpha=0.5))

    with pytest.raises(ValueError, match="does not match"):
        run_klent_training(mismatched, iterations=2)


@requires_core
def test_klent_fit_reduces_loss_over_epochs(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        fit_epochs=10,
        min_transitions=8,
        max_games_per_iteration=64,
        learning_rate=1e-3,
    )
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(20260522)
        summaries = run_klent_training(config, iterations=1)
    finally:
        torch.random.set_rng_state(rng_state)

    epoch_losses = summaries[0].epoch_losses
    assert len(epoch_losses) == 10
    assert min(epoch_losses[1:]) < epoch_losses[0]


@requires_core
@requires_onnx
def test_run_klent_training_publishes_onnx_pointer(tmp_path: Path) -> None:
    config = make_config(tmp_path, export_onnx=True, onnx_precision="fp32")

    summaries = run_klent_training(config, iterations=1)

    pointer = load_klent_onnx_pointer(config.work_dir)
    manifest = load_klent_onnx_manifest(pointer.manifest_path)
    assert pointer.model_version == 1
    assert summaries[0].onnx_version_dir is not None
    assert Path(pointer.actor_path).exists()
    assert Path(pointer.eval_path).exists()
    assert {record.kind for record in manifest.exports} == {"eval", "actor"}
    assert all(record.parity_passed for record in manifest.exports)


@requires_core
@requires_onnx
def test_run_klent_training_with_rust_actor_two_iterations(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        use_rust_actor=True,
        export_onnx=False,
        min_transitions=1,
        max_games_per_iteration=1,
        rust_self_play_batch_size=1,
        fit_epochs=1,
        batch_size=8,
    )

    summaries = run_klent_training(config, iterations=2)

    assert [summary.iteration for summary in summaries] == [0, 1]
    for summary in summaries:
        store, metadata = load_klent_shard(summary.shard_path)
        assert metadata.iteration == summary.iteration
        assert metadata.model_version == summary.iteration
        assert store.lambda_returns_present is not None
        assert bool(store.lambda_returns_present.all())
        assert store.model_versions.tolist() == [summary.iteration] * len(store)
    latest = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest.iteration == 2


def test_load_klent_checkpoint_rejects_state_value_checkpoint(tmp_path: Path) -> None:
    from great_kingdom_ai.klent.shards import KLENT_ALGORITHM
    from great_kingdom_ai.model import ModelConfig

    path = tmp_path / "state-value.pt"
    torch.save(
        {
            "algorithm": KLENT_ALGORITHM,
            "iteration": 1,
            "total_steps": 0,
            "model_preset": "small",
            "model_config": asdict(ModelConfig()),
            "klent_config": asdict(KlentConfig()),
            "model_state": {},
            "optimizer": "adamw",
            "optimizer_state": {},
            "last_shard": None,
        },
        path,
    )

    with pytest.raises(ValueError, match="action_value_head"):
        load_klent_checkpoint(path)


def test_load_klent_checkpoint_rejects_foreign_algorithm(tmp_path: Path) -> None:
    path = tmp_path / "gumbel.pt"
    torch.save({"algorithm": "gumbel"}, path)

    with pytest.raises(ValueError, match="algorithm"):
        load_klent_checkpoint(path)


def test_save_klent_checkpoint_keeps_previous_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        OptimizerConfig(learning_rate=1e-2),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=1,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    path = tmp_path / "iteration-0001.pt"
    save_klent_checkpoint(state, path)

    def failing_save(obj: object, f: object, *args: object, **kwargs: object) -> None:
        Path(f).write_bytes(b"partial checkpoint")  # type: ignore[arg-type]
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", failing_save)

    with pytest.raises(RuntimeError, match="disk full"):
        save_klent_checkpoint(state, path)

    assert not (tmp_path / ".iteration-0001.pt.tmp").exists()
    assert load_klent_checkpoint(path).iteration == 1


def test_warm_start_klent_model_copies_backbone_and_initializes_q_head(
    tmp_path: Path,
) -> None:
    from great_kingdom_ai.features import BOARD_SIZE, FEATURE_CHANNELS
    from great_kingdom_ai.model import create_model

    source = create_model("small")
    path = tmp_path / "source.pt"
    torch.save(
        {
            "model_config": asdict(source.config),
            "model_state": source.state_dict(),
        },
        path,
    )

    warm = warm_start_klent_model(path, model_preset="small")
    warm.eval()

    assert warm.has_action_value_head
    source_state = source.state_dict()
    warm_state = warm.state_dict()
    assert torch.equal(source_state["stem.0.weight"], warm_state["stem.0.weight"])
    assert torch.equal(source_state["policy_pass.2.weight"], warm_state["policy_pass.2.weight"])

    with torch.no_grad():
        logits, q_values = warm.forward_q(
            torch.zeros((2, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE))
        )
    assert logits.shape[0] == 2
    assert q_values.shape[0] == 2


class _FixedDataset:
    """Minimal KLENT dataset stand-in for direct fit tests."""

    def __init__(self, size: int) -> None:
        rng = np.random.default_rng(0)
        self._features = rng.normal(size=(size, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE))
        self._features = self._features.astype(np.float32)
        self._features[:, 4, :, :] = 1.0
        self._policies = np.full((size, ACTION_SPACE), 1.0 / ACTION_SPACE, dtype=np.float32)
        self._values = np.linspace(-1.0, 1.0, size).astype(np.float32)
        self._weights = np.ones((size,), dtype=np.float32)
        self._masks = np.ones((size, ACTION_SPACE), dtype=np.bool_)
        self._actions = np.arange(size, dtype=np.int64) % ACTION_SPACE

    def __len__(self) -> int:
        return self._features.shape[0]

    def arrays_for_indexes(self, indexes: np.ndarray) -> TrajectoryArrayBatch:
        return TrajectoryArrayBatch(
            indexes=np.asarray(indexes, dtype=np.int64),
            features=self._features[indexes],
            policies=self._policies[indexes],
            values=self._values[indexes],
            sample_weights=self._weights[indexes],
            legal_masks=self._masks[indexes],
            actions=self._actions[indexes],
        )


def test_fit_klent_model_applies_amp_autocast_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    recorded: list[bool] = []
    monkeypatch.setattr(
        trainer_module,
        "_cuda_amp_enabled",
        lambda torch_, device, enabled: bool(enabled),
    )

    def fake_autocast(torch_: object, *, enabled: bool) -> object:
        recorded.append(enabled)
        return nullcontext()

    monkeypatch.setattr(trainer_module, "_autocast_context", fake_autocast)
    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        OptimizerConfig(learning_rate=1e-2),
    )
    dataset = _FixedDataset(4)
    config = make_config(tmp_path, amp=True, batch_size=2, fit_epochs=1)

    losses, steps = fit_klent_model(
        model,
        dataset,  # type: ignore[arg-type]
        config,
        optimizer,
        start_steps=0,
        iteration=0,
    )

    assert recorded == [True, True]
    assert steps == 2
    assert all(np.isfinite(loss) for loss in losses)


@requires_core
def test_collection_fails_when_min_transitions_is_not_reached(tmp_path: Path) -> None:
    model = create_model("small_klent")
    config = make_config(
        tmp_path,
        min_transitions=10**6,
        max_games_per_iteration=1,
        fit_epochs=1,
    )

    with pytest.raises(RuntimeError, match="min_transitions"):
        _collect_with_python_actor(model, config, 0)


@requires_core
@requires_onnx
def test_rust_collection_fails_when_min_transitions_is_not_reached(tmp_path: Path) -> None:
    from great_kingdom_ai.klent.export import export_klent_checkpoint_to_onnx
    from great_kingdom_ai.klent.trainer import _collect_with_rust_actor

    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        OptimizerConfig(learning_rate=1e-2),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=0,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    checkpoint = trainer_module.save_klent_checkpoint(state, tmp_path / "actor-source.pt")
    actor_path = tmp_path / "actor.onnx"
    export_klent_checkpoint_to_onnx(checkpoint, actor_path, kind="actor")
    config = make_config(
        tmp_path,
        actor_onnx_path=actor_path,
        min_transitions=10**6,
        max_games_per_iteration=1,
        rust_self_play_batch_size=1,
        fit_epochs=1,
    )

    with pytest.raises(RuntimeError, match="min_transitions"):
        _collect_with_rust_actor(state, config, 0)


@requires_onnx
def test_actor_path_override_only_applies_to_first_iteration(tmp_path: Path) -> None:
    from great_kingdom_ai.klent.export import export_klent_checkpoint_to_onnx

    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        OptimizerConfig(learning_rate=1e-2),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=0,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    provided_actor = tmp_path / "provided-actor.onnx"
    source = save_klent_checkpoint(state, tmp_path / "provided-source.pt")
    export_klent_checkpoint_to_onnx(source, provided_actor, kind="actor")
    config = make_config(tmp_path, actor_onnx_path=provided_actor)
    (config.work_dir / "onnx").mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []

    def fake_export(checkpoint: object, output: Path, **kwargs: object) -> None:
        exported.append(Path(output))
        Path(output).write_bytes(b"placeholder")

    first = _actor_onnx_for_iteration(state, config, 0, fake_export)
    second = _actor_onnx_for_iteration(state, config, 1, fake_export)

    assert first == provided_actor
    assert second.name == "actor-source-0001.onnx"
    assert second != provided_actor
    assert exported == [second]


@requires_onnx
def test_actor_path_override_mismatch_is_rejected(tmp_path: Path) -> None:
    from great_kingdom_ai.klent.export import export_klent_checkpoint_to_onnx

    model = create_model("small_klent")
    optimizer = create_optimizer(
        torch,
        model,
        OptimizerConfig(learning_rate=1e-2),
    )
    state = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=0,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    other = create_model("small_klent")
    other_optimizer = create_optimizer(
        torch,
        other,
        OptimizerConfig(learning_rate=1e-2),
    )
    other_state = KlentTrainState(
        model=other,
        optimizer=other_optimizer,
        iteration=0,
        total_steps=0,
        klent_config=KlentConfig(),
        model_preset="small_klent",
    )
    other_checkpoint = save_klent_checkpoint(other_state, tmp_path / "other.pt")
    mismatched_actor = tmp_path / "mismatched-actor.onnx"
    export_klent_checkpoint_to_onnx(other_checkpoint, mismatched_actor, kind="actor")
    config = make_config(tmp_path, actor_onnx_path=mismatched_actor)
    exported: list[Path] = []

    def fake_export(checkpoint: object, output: Path, **kwargs: object) -> None:
        exported.append(Path(output))

    with pytest.raises(RuntimeError, match="actor_onnx_path"):
        _actor_onnx_for_iteration(state, config, 0, fake_export)

    assert exported == []


@requires_core
@requires_onnx
def test_resume_republishes_after_export_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, export_onnx=True, onnx_precision="fp32")
    real_publish = trainer_module.publish_klent_onnx_artifacts
    calls = {"count": 0}

    def flaky_publish(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("injected export failure")
        return real_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(trainer_module, "publish_klent_onnx_artifacts", flaky_publish)

    with pytest.raises(RuntimeError, match="injected export failure"):
        run_klent_training(config, iterations=1)

    latest = config.work_dir / "checkpoints" / "latest.pt"
    assert not latest.exists()
    with pytest.raises(ValueError, match="pointer is missing"):
        load_klent_onnx_pointer(config.work_dir)

    resumed = run_klent_training(config, iterations=1)

    assert resumed == []
    pointer = load_klent_onnx_pointer(config.work_dir)
    assert pointer.model_version == 1
    assert latest.exists()
    assert calls["count"] == 2


@requires_core
@requires_onnx
def test_resume_republishes_onnx_when_pointer_belongs_to_previous_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, export_onnx=True, onnx_precision="fp32")
    # 1. First run completes iteration 1 and publishes ONNX
    run_klent_training(config, iterations=1)
    pointer_run1 = load_klent_onnx_pointer(config.work_dir)
    manifest_run1 = load_klent_onnx_manifest(pointer_run1.manifest_path)
    assert pointer_run1.model_version == 1
    assert pointer_run1.run_id != ""
    assert manifest_run1.run_id == pointer_run1.run_id

    # 2. Fresh run (resume=False) starts, trains iteration 1, but export fails
    real_publish = trainer_module.publish_klent_onnx_artifacts
    publish_calls = {"count": 0}

    def failing_publish(*args: object, **kwargs: object) -> object:
        publish_calls["count"] += 1
        raise RuntimeError("injected export failure in fresh run")

    monkeypatch.setattr(trainer_module, "publish_klent_onnx_artifacts", failing_publish)

    with pytest.raises(RuntimeError, match="injected export failure in fresh run"):
        run_klent_training(config, iterations=1, resume=False)

    # The ONNX pointer in work_dir is still pointing to Run 1
    pointer_during_failure = load_klent_onnx_pointer(config.work_dir)
    assert pointer_during_failure.run_id == pointer_run1.run_id
    assert publish_calls["count"] == 1

    # 3. Resume the fresh run. It must NOT treat Run 1's version 1 as completed.
    monkeypatch.setattr(trainer_module, "publish_klent_onnx_artifacts", real_publish)
    resumed = run_klent_training(config, iterations=1, resume=True)

    assert resumed == []
    pointer_run2 = load_klent_onnx_pointer(config.work_dir)
    manifest_run2 = load_klent_onnx_manifest(pointer_run2.manifest_path)
    assert pointer_run2.model_version == 1
    assert pointer_run2.run_id != pointer_run1.run_id
    assert manifest_run2.run_id == pointer_run2.run_id


@requires_core
def test_resume_prefers_latest_checkpoint_over_older_iteration(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    # 1. Run 2 iterations
    run_klent_training(config, iterations=2)
    latest_before = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest_before.iteration == 2

    # 2. Delete iteration-0002.pt, leaving iteration-0001.pt and latest.pt
    iter2_path = config.work_dir / "checkpoints" / "iteration-0002.pt"
    assert iter2_path.exists()
    iter2_path.unlink()

    # 3. Resume with iterations=2. Must select latest.pt (iteration 2) instead of iteration-0001.pt
    resumed = run_klent_training(config, iterations=2, resume=True)

    assert resumed == []
    latest_after = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest_after.iteration == 2
    assert iter2_path.exists()
    restored_iter2 = load_klent_checkpoint(iter2_path)
    assert restored_iter2.iteration == 2

    # 4. Continuing to iteration 3 trains only iteration 2 -> 3
    continued = run_klent_training(config, iterations=3, resume=True)
    assert [summary.iteration for summary in continued] == [2]
    latest_iter3 = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest_iter3.iteration == 3


@requires_core
def test_resume_repairs_corrupt_iteration_from_latest_checkpoint(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_klent_training(config, iterations=2)
    iter2_path = config.work_dir / "checkpoints" / "iteration-0002.pt"
    iter2_path.write_bytes(b"corrupt bytes")

    resumed = run_klent_training(config, iterations=2, resume=True)
    assert resumed == []
    latest = load_klent_checkpoint(config.work_dir / "checkpoints" / "latest.pt")
    assert latest.iteration == 2
    repaired_iter2 = load_klent_checkpoint(iter2_path)
    assert repaired_iter2.iteration == 2


@requires_core
def test_resume_migrates_legacy_checkpoints_without_run_id(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    model = create_model(config.model_preset)
    optimizer = create_optimizer(
        torch,
        model,
        OptimizerConfig(
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
    )
    # 1. Manually create legacy checkpoints with run_id="" (simulating older version)
    checkpoints_dir = config.work_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    state_iter1 = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=1,
        total_steps=8,
        klent_config=config.klent,
        model_preset=config.model_preset,
        run_id="",
    )
    save_klent_checkpoint(state_iter1, checkpoints_dir / "iteration-0001.pt")

    state_iter2 = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=2,
        total_steps=17,
        klent_config=config.klent,
        model_preset=config.model_preset,
        run_id="",
    )
    save_klent_checkpoint(state_iter2, checkpoints_dir / "iteration-0002.pt")
    save_klent_checkpoint(state_iter2, checkpoints_dir / "latest.pt")

    marker = checkpoints_dir / "run.json"
    assert not marker.exists()

    # 2. Resume with iterations=2. Must NOT restart from iteration 0 / step 0.
    resumed = run_klent_training(config, iterations=2, resume=True)

    assert resumed == []
    latest = load_klent_checkpoint(checkpoints_dir / "latest.pt")
    assert latest.iteration == 2
    assert latest.total_steps == 17
    assert latest.run_id != ""

    assert marker.exists()
    marker_data = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_data["run_id"] == latest.run_id

    iter2 = load_klent_checkpoint(checkpoints_dir / "iteration-0002.pt")
    assert iter2.iteration == 2
    assert iter2.total_steps == 17
    assert iter2.run_id == latest.run_id

    iter1 = load_klent_checkpoint(checkpoints_dir / "iteration-0001.pt")
    assert iter1.run_id == latest.run_id

    # 3. Continuing to iteration 3 starts from iteration 2 and step 17
    continued = run_klent_training(config, iterations=3, resume=True)
    assert [summary.iteration for summary in continued] == [2]
    latest_iter3 = load_klent_checkpoint(checkpoints_dir / "latest.pt")
    assert latest_iter3.iteration == 3
    assert latest_iter3.total_steps > 17
    assert latest_iter3.run_id == latest.run_id


@requires_core
def test_resume_false_interrupted_does_not_adopt_legacy_checkpoints(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    model = create_model(config.model_preset)
    optimizer = create_optimizer(
        torch,
        model,
        OptimizerConfig(
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
    )
    # 1. Manually create legacy checkpoints with run_id="" (iteration 2, 17 steps)
    checkpoints_dir = config.work_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    state_iter2 = KlentTrainState(
        model=model,
        optimizer=optimizer,
        iteration=2,
        total_steps=17,
        klent_config=config.klent,
        model_preset=config.model_preset,
        run_id="",
    )
    save_klent_checkpoint(state_iter2, checkpoints_dir / "iteration-0002.pt")
    save_klent_checkpoint(state_iter2, checkpoints_dir / "latest.pt")

    # 2. Start a fresh run with resume=False, interrupted before saving first checkpoint
    with unittest.mock.patch(
        "great_kingdom_ai.klent.trainer.fit_klent_model_detailed",
        side_effect=RuntimeError("interrupted before checkpoint"),
    ):
        with pytest.raises(RuntimeError, match="interrupted before checkpoint"):
            run_klent_training(config, iterations=1, resume=False)

    marker = checkpoints_dir / "run.json"
    assert marker.exists()
    marker_data = json.loads(marker.read_text(encoding="utf-8"))
    fresh_run_id = marker_data["run_id"]
    assert fresh_run_id != ""
    assert not marker_data.get("allow_legacy", False)

    # 3. Resuming with resume=True must NOT adopt legacy checkpoints;
    # it must start fresh at iteration 0 and execute iteration 0.
    resumed = run_klent_training(config, iterations=1, resume=True)
    assert [summary.iteration for summary in resumed] == [0]

    latest = load_klent_checkpoint(checkpoints_dir / "latest.pt")
    assert latest.iteration == 1
    assert latest.run_id == fresh_run_id
    assert latest.total_steps > 0

    # The legacy iteration 2 checkpoint was neither adopted nor migrated
    legacy_iter2 = load_klent_checkpoint(checkpoints_dir / "iteration-0002.pt")
    assert legacy_iter2.iteration == 2
    assert legacy_iter2.total_steps == 17
    assert legacy_iter2.run_id == ""

def test_resume_lr_override_preserves_adam_state(tmp_path: Path) -> None:
    from dataclasses import replace

    model = create_model('small_klent')
    optimizer = create_optimizer(torch, model, OptimizerConfig(learning_rate=3e-4))
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = KlentTrainState(model=model, optimizer=optimizer, iteration=7,
                            total_steps=42, klent_config=KlentConfig(),
                            model_preset='small_klent')
    path = save_klent_checkpoint(state, tmp_path / 'source.pt')
    config = make_config(tmp_path, learning_rate=1e-4)
    normal = trainer_module._load_state(path, config)
    assert normal.optimizer.param_groups[0]['lr'] == 3e-4
    overridden = trainer_module._load_state(path, replace(config, override_learning_rate=True))
    assert overridden.optimizer.param_groups[0]['lr'] == 1e-4
    assert overridden.iteration == 7
    assert overridden.total_steps == 42
    original_stats = next(iter(optimizer.state.values()))
    restored_stats = next(iter(overridden.optimizer.state.values()))
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(original_stats[key], restored_stats[key])
    saved = save_klent_checkpoint(overridden, tmp_path / 'next.pt')
    assert load_klent_checkpoint(saved).optimizer.param_groups[0]['lr'] == 1e-4
