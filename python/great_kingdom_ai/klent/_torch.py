"""Lazy torch import and AMP helpers for the standalone KLENT package."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for KLENT target and loss computation") from exc
    return torch


def _cuda_amp_enabled(
    torch: Any,
    device: Any | str | None,
    *,
    enabled: bool,
) -> bool:
    return bool(
        enabled
        and device is not None
        and str(device).startswith("cuda")
        and torch.cuda.is_available()
    )


def _autocast_context(torch: Any, *, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    return torch.amp.autocast("cuda", enabled=True)


__all__ = ["_autocast_context", "_cuda_amp_enabled", "_import_torch"]
