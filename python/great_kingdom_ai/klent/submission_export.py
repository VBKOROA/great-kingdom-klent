"""Export a frozen KLENT submission candidate with CPU FP32 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from great_kingdom_ai.replay.persistence import copy_file_atomic


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def export_submission(
    source: Path,
    destination: Path,
    *,
    tolerance: float = 1e-4,
    batch_sizes: tuple[int, ...] = (1, 3, 8),
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    """Create a new candidate directory; preserve diagnostics on parity failure."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch sizes must be positive")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"Candidate already exists; choose a new directory: {destination}")

    import onnx
    import onnxruntime as ort
    import torch

    from great_kingdom_ai.klent.checkpoint import read_klent_checkpoint_metadata
    from great_kingdom_ai.klent.export import (
        compare_klent_checkpoint_to_onnx,
        export_klent_checkpoint_to_onnx,
        onnx_output_names,
    )

    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = destination / "source.pt"
    candidate = destination / "model.pending.onnx"
    output = destination / "model.onnx"
    report_path = destination / "export-report.json"
    report: dict[str, Any] = {
        "source": str(source),
        "checkpoint": str(checkpoint),
        "output": str(candidate),
        "kind": "eval",
        "precision": "fp32",
        "status": "running",
        "tolerance": tolerance,
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "parity_checks": [],
    }
    try:
        print(f"Freezing checkpoint: {source} -> {checkpoint}", flush=True)
        copy_file_atomic(source, checkpoint)
        report["checkpoint_sha256"] = _sha256(checkpoint)
        metadata = read_klent_checkpoint_metadata(checkpoint)
        report.update(
            iteration=metadata.iteration, total_steps=metadata.total_steps, run_id=metadata.run_id
        )
        print("Exporting eval ONNX (CPU FP32)...", flush=True)
        export = export_klent_checkpoint_to_onnx(
            checkpoint,
            candidate,
            kind="eval",
            device="cpu",
            precision="fp32",
        )
        report["opset"] = export.opset_version
        onnx.checker.check_model(str(candidate))
        if onnx_output_names(candidate) != ["policy_logits", "value"]:
            raise RuntimeError("Unexpected output contract")
        report["onnx_sha256"] = _sha256(candidate)
        for batch_size in batch_sizes:
            for seed in seeds:
                result = compare_klent_checkpoint_to_onnx(
                    checkpoint,
                    candidate,
                    kind="eval",
                    batch_size=batch_size,
                    seed=seed,
                    tolerance=tolerance,
                )
                check = {**result.to_json_dict(), "batch_size": batch_size, "seed": seed}
                report["parity_checks"].append(check)
                print(json.dumps(check), flush=True)
        if not all(check["passed"] for check in report["parity_checks"]):
            raise RuntimeError(f"Parity failed; inspect {report_path}")
        # Write the report before promotion, so report I/O failure leaves a pending model.
        report.update(status="passed", output=str(output))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        candidate.replace(output)
    except Exception as exc:
        report.update(status="failed", output=str(candidate), error=str(exc))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    print(f"Export complete: {output}\nReport: {report_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Source KLENT .pt file")
    parser.add_argument("--output-dir", type=Path, required=True, help="New candidate directory")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Maximum absolute CPU FP32 error (default: 1e-4)",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 3, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)
    try:
        export_submission(
            args.checkpoint,
            args.output_dir,
            tolerance=args.tolerance,
            batch_sizes=tuple(args.batch_sizes),
            seeds=tuple(args.seeds),
        )
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
