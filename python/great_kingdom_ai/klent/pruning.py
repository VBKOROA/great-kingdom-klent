"""Retention for completed KLENT shards, checkpoints and ONNX exports.

Only artifacts of iterations that ``latest.pt`` has already superseded are
selected. ``latest.pt``, ``run.json``, the published ONNX pointer and snapshots
are never touched, and the in-progress iteration (including its ``actor-source``
files) is always preserved.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from great_kingdom_ai.klent.checkpoint import read_klent_checkpoint_metadata
from great_kingdom_ai.klent.publish import (
    KLENT_ONNX_MANIFEST_NAME,
    KLENT_ONNX_POINTER_NAME,
)
from great_kingdom_ai.klent.shards import KlentShardMetadata, shard_metadata_path

_CHECKPOINT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("iteration", re.compile(r"iteration-(\d+)\.pt")),
    ("actor-source", re.compile(r"actor-source-(\d+)\.pt")),
)
_ONNX_VERSION_PATTERN = re.compile(r"version-(\d+)")
_ONNX_ACTOR_SOURCE_PATTERN = re.compile(r"actor-source-(\d+)\.onnx")


@dataclass(frozen=True)
class ShardPruneItem:
    shard: Path
    metadata: Path
    iteration: int
    size_bytes: int
    signatures: tuple[tuple[int, int, int], tuple[int, int, int]]


@dataclass(frozen=True)
class CheckpointPruneItem:
    path: Path
    iteration: int
    kind: str
    size_bytes: int
    signature: tuple[int, int, int]


@dataclass(frozen=True)
class OnnxPruneItem:
    path: Path
    model_version: int
    kind: str
    size_bytes: int
    signature: tuple[int, int, int]


def _signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _completed_state(work_dir: Path) -> tuple[str, int]:
    marker = work_dir / "checkpoints" / "run.json"
    latest = work_dir / "checkpoints" / "latest.pt"
    if marker.is_symlink() or latest.is_symlink():
        raise ValueError("run marker and latest checkpoint must not be symlinks")
    run_id = json.loads(marker.read_text())["run_id"]
    state = read_klent_checkpoint_metadata(latest)
    if not run_id or state.run_id != run_id:
        raise ValueError("latest checkpoint must belong to the active, non-legacy run")
    return state.run_id, state.iteration


def prune_completed_shards(
    work_dir: Path,
    *,
    keep_shards: int = 2,
    min_age_seconds: float = 600,
    delete: bool = False,
) -> list[ShardPruneItem]:
    """Plan or remove older completed shard pairs, failing closed on invalid state.

    Run this alongside one continuing trainer. Stop the worker before resetting
    or restoring the work directory. Missing/corrupt metadata is left untouched.
    """
    if keep_shards < 0:
        raise ValueError("keep_shards must be non-negative")
    if not math.isfinite(min_age_seconds) or min_age_seconds < 0:
        raise ValueError("min_age_seconds must be finite and non-negative")
    root = work_dir.resolve(strict=True)
    for directory in (root / "iterations", root / "checkpoints"):
        if directory.is_symlink():
            raise ValueError(f"refusing symlink directory: {directory}")
    initial_state = _completed_state(root)
    completed_iteration = initial_state[1]
    pairs: list[ShardPruneItem] = []
    cutoff = time.time() - min_age_seconds
    for shard in (root / "iterations").glob("iteration-*.npz"):
        match = re.fullmatch(r"iteration-(\d+)\.npz", shard.name)
        if match is None or int(match[1]) >= completed_iteration:
            continue
        metadata_path = shard_metadata_path(shard)
        if shard.is_symlink() or metadata_path.is_symlink():
            continue
        try:
            metadata = KlentShardMetadata.from_dict(json.loads(metadata_path.read_text()))
            if metadata.iteration != int(match[1]) or metadata.model_version != int(match[1]):
                continue
            signatures = (_signature(shard), _signature(metadata_path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        pairs.append(
            ShardPruneItem(
                shard,
                metadata_path,
                int(match[1]),
                signatures[0][1] + signatures[1][1],
                signatures,
            )
        )
    pairs.sort(key=lambda item: item.iteration, reverse=True)
    selected = [
        item
        for item in pairs[keep_shards:]
        if max(signature[2] for signature in item.signatures) / 1e9 <= cutoff
    ]
    if not delete:
        return selected
    # Reload before mutations: a new run or a changed checkpoint invalidates this plan.
    if _completed_state(root) != initial_state:
        return []
    removed: list[ShardPruneItem] = []
    for item in selected:
        try:
            if item.shard.is_symlink() or item.metadata.is_symlink():
                continue
            if (_signature(item.shard), _signature(item.metadata)) != item.signatures:
                continue
            item.shard.unlink()
            item.metadata.unlink(missing_ok=True)
            removed.append(item)
        except FileNotFoundError:
            # The trainer's keep_shards=False cleanup can race this worker.
            continue
    return removed


def prune_completed_checkpoints(
    work_dir: Path,
    *,
    keep_iterations: int = 2,
    keep_actor_sources: int = 1,
    min_age_seconds: float = 600,
    delete: bool = False,
) -> list[CheckpointPruneItem]:
    """Plan or remove older completed iteration/actor-source checkpoints.

    ``latest.pt`` and ``run.json`` are always preserved, as is the iteration the
    active checkpoint belongs to, so a restart can still resume. Among the
    superseded checkpoints, the newest ``keep_iterations`` recovery checkpoints
    and ``keep_actor_sources`` actor sources are retained per kind. Foreign or
    missing run identifiers are not verified here because reading every
    candidate checkpoint is expensive; run selection already filters by run id.
    """
    if keep_iterations < 0:
        raise ValueError("keep_iterations must be non-negative")
    if keep_actor_sources < 0:
        raise ValueError("keep_actor_sources must be non-negative")
    if not math.isfinite(min_age_seconds) or min_age_seconds < 0:
        raise ValueError("min_age_seconds must be finite and non-negative")
    root = work_dir.resolve(strict=True)
    checkpoints = root / "checkpoints"
    if checkpoints.is_symlink():
        raise ValueError(f"refusing symlink directory: {checkpoints}")
    initial_state = _completed_state(root)
    completed_iteration = initial_state[1]
    cutoff = time.time() - min_age_seconds
    groups: dict[str, list[CheckpointPruneItem]] = {
        kind: [] for kind in ("iteration", "actor-source")
    }
    for path in checkpoints.glob("*.pt"):
        if path.is_symlink():
            continue
        for kind, pattern in _CHECKPOINT_PATTERNS:
            match = pattern.fullmatch(path.name)
            if match is None or int(match[1]) >= completed_iteration:
                continue
            try:
                signature = _signature(path)
            except OSError:
                continue
            groups[kind].append(
                CheckpointPruneItem(
                    path,
                    int(match[1]),
                    kind,
                    signature[1],
                    signature,
                )
            )
    selected: list[CheckpointPruneItem] = []
    for kind, keep in (("iteration", keep_iterations), ("actor-source", keep_actor_sources)):
        items = sorted(groups[kind], key=lambda item: item.iteration, reverse=True)
        selected.extend(item for item in items[keep:] if item.signature[2] / 1e9 <= cutoff)
    selected.sort(key=lambda item: (item.kind, -item.iteration))
    if not delete:
        return selected
    # Reload before mutations: a new run or a changed checkpoint invalidates this plan.
    if _completed_state(root) != initial_state:
        return []
    removed: list[CheckpointPruneItem] = []
    for item in selected:
        try:
            if item.path.is_symlink() or _signature(item.path) != item.signature:
                continue
            item.path.unlink()
            removed.append(item)
        except FileNotFoundError:
            continue
    return removed


def prune_completed_onnx(
    work_dir: Path,
    *,
    keep_versions: int = 2,
    keep_actor_sources: int = 1,
    min_age_seconds: float = 600,
    delete: bool = False,
) -> list[OnnxPruneItem]:
    """Plan or remove older completed ``onnx/version-*`` and actor-source exports.

    The published pointer (``onnx/current.json``) and the iteration ``latest.pt``
    belongs to are always preserved. Version directories are validated against
    their manifest before removal so partially written exports are left alone.
    """
    if keep_versions < 0:
        raise ValueError("keep_versions must be non-negative")
    if keep_actor_sources < 0:
        raise ValueError("keep_actor_sources must be non-negative")
    if not math.isfinite(min_age_seconds) or min_age_seconds < 0:
        raise ValueError("min_age_seconds must be finite and non-negative")
    root = work_dir.resolve(strict=True)
    onnx_root = root / "onnx"
    if onnx_root.is_symlink():
        raise ValueError(f"refusing symlink directory: {onnx_root}")
    initial_state = _completed_state(root)
    completed_iteration = initial_state[1]
    cutoff = time.time() - min_age_seconds
    current_version = _current_onnx_version(onnx_root)
    groups: dict[str, list[OnnxPruneItem]] = {"version": [], "actor-source": []}
    for path in onnx_root.glob("version-*"):
        match = _ONNX_VERSION_PATTERN.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            continue
        version = int(match[1])
        if version >= completed_iteration or version == current_version:
            continue
        if _read_manifest_version(path / KLENT_ONNX_MANIFEST_NAME) != version:
            continue
        try:
            signature = _signature(path)
        except OSError:
            continue
        groups["version"].append(
            OnnxPruneItem(path, version, "version", _directory_size(path), signature)
        )
    for path in onnx_root.glob("actor-source-*.onnx"):
        match = _ONNX_ACTOR_SOURCE_PATTERN.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            continue
        version = int(match[1])
        if version >= completed_iteration:
            continue
        try:
            signature = _signature(path)
        except OSError:
            continue
        groups["actor-source"].append(
            OnnxPruneItem(path, version, "actor-source", signature[1], signature)
        )
    selected: list[OnnxPruneItem] = []
    for kind, keep in (("version", keep_versions), ("actor-source", keep_actor_sources)):
        items = sorted(groups[kind], key=lambda item: item.model_version, reverse=True)
        selected.extend(item for item in items[keep:] if item.signature[2] / 1e9 <= cutoff)
    selected.sort(key=lambda item: (item.kind, -item.model_version))
    if not delete:
        return selected
    # Reload before mutations: a new run, pointer move or changed export invalidates this plan.
    if _completed_state(root) != initial_state:
        return []
    removed: list[OnnxPruneItem] = []
    for item in selected:
        try:
            if item.path.is_symlink() or _signature(item.path) != item.signature:
                continue
            if item.kind == "version":
                if _current_onnx_version(onnx_root) == item.model_version:
                    continue
                manifest = item.path / KLENT_ONNX_MANIFEST_NAME
                if _read_manifest_version(manifest) != item.model_version:
                    continue
                shutil.rmtree(item.path)
            else:
                item.path.unlink()
            removed.append(item)
        except FileNotFoundError:
            continue
    return removed


def _current_onnx_version(onnx_root: Path) -> int | None:
    pointer = onnx_root / KLENT_ONNX_POINTER_NAME
    if pointer.is_symlink() or not pointer.is_file():
        return None
    try:
        data = json.loads(pointer.read_text())
        return int(data["model_version"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _read_manifest_version(manifest_path: Path) -> int | None:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        return int(data["model_version"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


__all__ = [
    "CheckpointPruneItem",
    "OnnxPruneItem",
    "ShardPruneItem",
    "prune_completed_checkpoints",
    "prune_completed_onnx",
    "prune_completed_shards",
]
