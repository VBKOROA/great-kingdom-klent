"""Run once or periodically prune completed KLENT shards, checkpoints and ONNX.

Deletion is a dry-run unless ``--delete`` is passed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from great_kingdom_ai.klent.pruning import (
    prune_completed_checkpoints,
    prune_completed_onnx,
    prune_completed_shards,
)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _report(work_dir: Path, args: argparse.Namespace) -> None:
    shards = prune_completed_shards(
        work_dir,
        keep_shards=args.keep_shards,
        min_age_seconds=args.min_age_seconds,
        delete=args.delete,
    )
    checkpoints = prune_completed_checkpoints(
        work_dir,
        keep_iterations=args.keep_iterations,
        keep_actor_sources=args.keep_actor_sources,
        min_age_seconds=args.min_age_seconds,
        delete=args.delete,
    )
    onnx = prune_completed_onnx(
        work_dir,
        keep_versions=args.keep_onnx_versions,
        keep_actor_sources=args.keep_onnx_actor_sources,
        min_age_seconds=args.min_age_seconds,
        delete=args.delete,
    )
    shard_paths = [str(path) for item in shards for path in (item.shard, item.metadata)]
    checkpoint_paths = [str(item.path) for item in checkpoints]
    onnx_paths = [str(item.path) for item in onnx]
    print(
        json.dumps(
            {
                "event": "klent_pruning",
                "mode": "deleted" if args.delete else "dry_run",
                "shards": len(shards),
                "checkpoints": len(checkpoints),
                "onnx": len(onnx),
                "bytes": sum(item.size_bytes for item in shards)
                + sum(item.size_bytes for item in checkpoints)
                + sum(item.size_bytes for item in onnx),
                "paths": shard_paths + checkpoint_paths + onnx_paths,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--keep-shards", type=_non_negative_int, default=2)
    parser.add_argument("--keep-iterations", type=_non_negative_int, default=2)
    parser.add_argument("--keep-actor-sources", type=_non_negative_int, default=1)
    parser.add_argument("--keep-onnx-versions", type=_non_negative_int, default=2)
    parser.add_argument("--keep-onnx-actor-sources", type=_non_negative_int, default=1)
    parser.add_argument("--min-age-seconds", type=_non_negative_float, default=600)
    parser.add_argument("--interval-seconds", type=_positive_float, default=600)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            try:
                _report(args.work_dir, args)
            except Exception as error:
                if not args.loop:
                    raise
                # Includes an absent initial checkpoint and transient checkpoint read failures.
                print(
                    json.dumps(
                        {
                            "event": "klent_pruning_skipped",
                            "error": str(error),
                        }
                    ),
                    flush=True,
                )
            if not args.loop:
                break
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
