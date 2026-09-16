from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from great_kingdom_ai.klent.checkpoint import KlentCheckpointMetadata
from great_kingdom_ai.klent.pruning import (
    prune_completed_checkpoints,
    prune_completed_onnx,
    prune_completed_shards,
)
from great_kingdom_ai.klent.shards import (
    KlentShardMetadata,
    shard_metadata_path,
    write_klent_shard_metadata,
)
from great_kingdom_ai.klent.types import KlentConfig


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "iterations").mkdir()
    (tmp_path / "checkpoints/run.json").write_text(json.dumps({"run_id": "active"}))
    (tmp_path / "checkpoints/latest.pt").write_bytes(b"protected")
    monkeypatch.setattr(
        "great_kingdom_ai.klent.pruning.read_klent_checkpoint_metadata",
        lambda _: KlentCheckpointMetadata("active", 4, 10),
    )
    return tmp_path


def shard(work: Path, iteration: int) -> Path:
    path = work / "iterations" / f"iteration-{iteration:04d}.npz"
    path.write_bytes(b"shard")
    write_klent_shard_metadata(
        KlentShardMetadata.from_config(
            KlentConfig(),
            iteration=iteration,
            model_version=iteration,
            transitions=2,
            games=1,
        ),
        path,
    )
    for item in (path, shard_metadata_path(path)):
        os.utime(item, (1, 1))
    return path


def test_only_completed_pairs_pruned_and_artifacts_protected(work: Path) -> None:
    paths = [shard(work, i) for i in range(6)]
    snapshot = work / "checkpoints/snapshots/example.pt"
    snapshot.parent.mkdir()
    snapshot.write_bytes(b"snapshot")
    temporary = work / "iterations/iteration-0000.npz.tmp"
    temporary.write_bytes(b"pending")
    plan = prune_completed_shards(work, keep_shards=2)
    assert {item.iteration for item in plan} == {0, 1}
    assert all(path.exists() for path in paths)
    removed = prune_completed_shards(work, keep_shards=2, delete=True)
    assert {item.iteration for item in removed} == {0, 1}
    for path in paths[:2]:
        assert not path.exists()
        assert not shard_metadata_path(path).exists()
    assert all(path.exists() for path in paths[2:])
    assert snapshot.read_bytes() == b"snapshot"
    assert (work / "checkpoints/latest.pt").read_bytes() == b"protected"
    assert temporary.exists()


def test_zero_keep_preserves_unfinished_shards(work: Path) -> None:
    paths = [shard(work, i) for i in (0, 3, 4)]
    prune_completed_shards(work, keep_shards=0, delete=True)
    assert not paths[0].exists() and not paths[1].exists()
    assert paths[2].exists()


def test_recent_missing_and_foreign_metadata_are_preserved(work: Path) -> None:
    recent, missing, foreign = [shard(work, i) for i in range(3)]
    os.utime(recent, None)
    shard_metadata_path(missing).unlink()
    shard_metadata_path(foreign).write_text('{"algorithm": "gumbel"}')
    assert prune_completed_shards(work, keep_shards=0, delete=True) == []
    assert all(path.exists() for path in (recent, missing, foreign))


def test_run_mismatch_fails_closed(work: Path) -> None:
    path = shard(work, 0)
    (work / "checkpoints/run.json").write_text('{"run_id": "new"}')
    with pytest.raises(ValueError, match="active"):
        prune_completed_shards(work, keep_shards=0, delete=True)
    assert path.exists()


def test_state_change_before_delete_cancels_plan(
    work: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = shard(work, 0)
    states = iter([KlentCheckpointMetadata("active", 4), KlentCheckpointMetadata("active", 5)])
    monkeypatch.setattr(
        "great_kingdom_ai.klent.pruning.read_klent_checkpoint_metadata",
        lambda _: next(states),
    )
    assert prune_completed_shards(work, keep_shards=0, delete=True) == []
    assert path.exists()


def test_symlink_shard_is_not_removed(work: Path, tmp_path: Path) -> None:
    path = shard(work, 0)
    external = tmp_path / "external.npz"
    path.rename(external)
    path.symlink_to(external)
    assert prune_completed_shards(work, keep_shards=0, delete=True) == []
    assert path.is_symlink() and external.exists()


@pytest.mark.parametrize(
    "options", [{"keep_shards": -1}, {"min_age_seconds": -1}, {"min_age_seconds": float("nan")}]
)
def test_invalid_options_rejected(work: Path, options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        prune_completed_shards(work, **options)


def checkpoint(work: Path, name: str) -> Path:
    path = work / "checkpoints" / name
    path.write_bytes(b"checkpoint")
    os.utime(path, (1, 1))
    return path


def test_checkpoint_retention_protects_active_and_latest(work: Path) -> None:
    iterations = [checkpoint(work, f"iteration-{i:04d}.pt") for i in range(5)]
    actors = [checkpoint(work, f"actor-source-{i:04d}.pt") for i in range(5)]
    plan = prune_completed_checkpoints(work, keep_iterations=1, keep_actor_sources=1)
    assert {item.kind for item in plan} == {"iteration", "actor-source"}
    assert {item.iteration for item in plan if item.kind == "iteration"} == {0, 1, 2}
    assert {item.iteration for item in plan if item.kind == "actor-source"} == {0, 1, 2}
    removed = prune_completed_checkpoints(
        work, keep_iterations=1, keep_actor_sources=1, delete=True
    )
    assert {item.path for item in removed} == set(iterations[:3] + actors[:3])
    assert iterations[3].exists() and iterations[4].exists()
    assert actors[3].exists() and actors[4].exists()
    assert (work / "checkpoints/latest.pt").exists()
    assert (work / "checkpoints/run.json").exists()


def test_checkpoint_zero_keep_removes_all_completed(work: Path) -> None:
    completed = [checkpoint(work, "iteration-0000.pt"), checkpoint(work, "actor-source-0000.pt")]
    active = checkpoint(work, "iteration-0004.pt")
    removed = prune_completed_checkpoints(
        work, keep_iterations=0, keep_actor_sources=0, delete=True
    )
    assert {item.path for item in removed} == set(completed)
    assert all(not path.exists() for path in completed)
    assert active.exists()


def test_checkpoint_recent_and_symlink_preserved(work: Path, tmp_path: Path) -> None:
    recent = checkpoint(work, "iteration-0001.pt")
    os.utime(recent, None)
    external = tmp_path / "external.pt"
    external.write_bytes(b"external")
    link = work / "checkpoints/actor-source-0000.pt"
    link.symlink_to(external)
    assert prune_completed_checkpoints(work, keep_iterations=0, keep_actor_sources=0) == []
    removed = prune_completed_checkpoints(
        work, keep_iterations=0, keep_actor_sources=0, delete=True
    )
    assert removed == []
    assert recent.exists() and link.is_symlink() and external.exists()


def test_checkpoint_state_change_before_delete_cancels_plan(
    work: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = checkpoint(work, "iteration-0000.pt")
    states = iter([KlentCheckpointMetadata("active", 4), KlentCheckpointMetadata("active", 5)])
    monkeypatch.setattr(
        "great_kingdom_ai.klent.pruning.read_klent_checkpoint_metadata",
        lambda _: next(states),
    )
    assert (
        prune_completed_checkpoints(work, keep_iterations=0, keep_actor_sources=0, delete=True)
        == []
    )
    assert path.exists()


@pytest.mark.parametrize(
    "options",
    [
        {"keep_iterations": -1},
        {"keep_actor_sources": -1},
        {"min_age_seconds": -1},
        {"min_age_seconds": float("nan")},
    ],
)
def test_checkpoint_invalid_options_rejected(work: Path, options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        prune_completed_checkpoints(work, **options)


def onnx_version(work: Path, version: int) -> Path:
    directory = work / "onnx" / f"version-{version:05d}"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"model_version": version}))
    (directory / "actor.onnx").write_bytes(b"actor")
    (directory / "eval.onnx").write_bytes(b"eval")
    os.utime(directory, (1, 1))
    return directory


def onnx_actor(work: Path, version: int) -> Path:
    (work / "onnx").mkdir(exist_ok=True)
    path = work / "onnx" / f"actor-source-{version:04d}.onnx"
    path.write_bytes(b"actor-source")
    os.utime(path, (1, 1))
    return path


def test_onnx_retention_protects_pointer_and_active(work: Path) -> None:
    versions = [onnx_version(work, version) for version in range(5)]
    actors = [onnx_actor(work, version) for version in range(5)]
    (work / "onnx/current.json").write_text(json.dumps({"model_version": 3}))
    plan = prune_completed_onnx(work, keep_versions=1, keep_actor_sources=1)
    assert {item.model_version for item in plan if item.kind == "version"} == {0, 1}
    assert {item.model_version for item in plan if item.kind == "actor-source"} == {0, 1, 2}
    removed = prune_completed_onnx(work, keep_versions=1, keep_actor_sources=1, delete=True)
    assert {item.path for item in removed} == set(versions[:2] + actors[:3])
    assert versions[2].exists() and versions[3].exists() and versions[4].exists()
    assert actors[3].exists() and actors[4].exists()
    assert (work / "onnx/current.json").exists()


def test_onnx_without_manifest_and_symlink_preserved(work: Path, tmp_path: Path) -> None:
    complete = onnx_version(work, 0)
    partial = work / "onnx/version-00001"
    partial.mkdir()
    (partial / "actor.onnx").write_bytes(b"partial")
    os.utime(partial, (1, 1))
    external = tmp_path / "external.onnx"
    external.write_bytes(b"external")
    link = work / "onnx/actor-source-0000.onnx"
    link.symlink_to(external)
    removed = prune_completed_onnx(work, keep_versions=0, keep_actor_sources=0, delete=True)
    assert {item.path for item in removed} == {complete}
    assert not complete.exists()
    assert partial.exists() and link.is_symlink() and external.exists()


@pytest.mark.parametrize(
    "options",
    [
        {"keep_versions": -1},
        {"keep_actor_sources": -1},
        {"min_age_seconds": -1},
        {"min_age_seconds": float("nan")},
    ],
)
def test_onnx_invalid_options_rejected(work: Path, options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        prune_completed_onnx(work, **options)
