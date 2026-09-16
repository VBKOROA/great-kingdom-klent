"""Shared checkpoint helpers retained for ONNX export and evaluation.

The legacy training loop, CLI, and batch sampling APIs were removed with the
Gumbel async v2 decommission. Only checkpoint construction/loading helpers
remain, and KLENT owns its own training batch and optimizer utilities.
"""

from great_kingdom_ai.training.checkpoint import (
    TrainState,
    create_lr_scheduler,
    create_train_state,
    load_checkpoint,
    load_checkpoint_weights,
    save_checkpoint,
    summarize_checkpoint_optimizer_state,
    summarize_optimizer_state_dict,
    warm_start_terminal_board_head,
)
from great_kingdom_ai.training.config import TrainingConfig, load_training_config

__all__ = [
    "TrainState",
    "TrainingConfig",
    "create_lr_scheduler",
    "create_train_state",
    "load_checkpoint",
    "load_checkpoint_weights",
    "load_training_config",
    "save_checkpoint",
    "summarize_checkpoint_optimizer_state",
    "summarize_optimizer_state_dict",
    "warm_start_terminal_board_head",
]
