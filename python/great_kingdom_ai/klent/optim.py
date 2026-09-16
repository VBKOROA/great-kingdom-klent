"""KLENT optimizer configuration and construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class OptimizerConfig:
    """Lean optimizer settings owned by the KLENT training loop."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    nesterov: bool = False


def create_optimizer(torch: Any, model: Any, config: OptimizerConfig) -> Any:
    optimizer = config.optimizer.lower()
    if optimizer == "adamw":
        return cast(
            "Any",
            torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            ),
        )
    if optimizer == "sgd":
        if not math.isfinite(config.momentum) or config.momentum < 0.0:
            raise ValueError("momentum must be finite and non-negative")
        return cast(
            "Any",
            torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.nesterov,
            ),
        )
    raise ValueError("optimizer must be one of: adamw, sgd")


__all__ = ["OptimizerConfig", "create_optimizer"]
