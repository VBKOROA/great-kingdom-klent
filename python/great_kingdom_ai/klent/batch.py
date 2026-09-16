"""KLENT training batch containers and tensor conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from great_kingdom_ai.features import BOARD_CELLS, LEGAL_PLACE_FEATURE_CHANNEL, PASS_ACTION
from great_kingdom_ai.klent._torch import _import_torch

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TrainingBatch:
    """Tensor batch consumed by the KLENT policy and Q-value losses."""

    features: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    legal_mask: torch.Tensor
    sample_weight: torch.Tensor
    replay_indexes: np.ndarray | None = None
    action: torch.Tensor | None = None


@dataclass(frozen=True)
class TrainingArrays:
    """NumPy arrays sampled from the KLENT replay before tensor conversion."""

    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    sample_weights: np.ndarray
    legal_masks: np.ndarray | None = None
    indexes: np.ndarray | None = None
    actions: np.ndarray | None = None


def arrays_to_batch(
    arrays: TrainingArrays,
    *,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
) -> TrainingBatch:
    torch = _import_torch()
    features = np.ascontiguousarray(arrays.features, dtype=np.float32)
    policies = np.ascontiguousarray(arrays.policies, dtype=np.float32)
    values = np.ascontiguousarray(arrays.values, dtype=np.float32)
    sample_weights = np.ascontiguousarray(arrays.sample_weights, dtype=np.float32)
    if features.shape[0] == 0:
        raise ValueError("training batch must contain at least one sample")
    legal_masks = (
        _legal_masks_from_features(features)
        if arrays.legal_masks is None
        else np.ascontiguousarray(arrays.legal_masks, dtype=np.bool_)
    )
    if legal_masks.shape != policies.shape:
        raise ValueError("legal_masks shape must match policies shape")
    actions = _actions_from_arrays(arrays)
    return TrainingBatch(
        features=_tensor_from_numpy(torch, features, pin_memory=pin_memory).to(device=device),
        policy=_tensor_from_numpy(torch, policies, pin_memory=pin_memory).to(device=device),
        value=_tensor_from_numpy(torch, values, pin_memory=pin_memory).to(device=device),
        legal_mask=_tensor_from_numpy(torch, legal_masks, pin_memory=pin_memory).to(
            device=device
        ),
        sample_weight=_tensor_from_numpy(torch, sample_weights, pin_memory=pin_memory).to(
            device=device
        ),
        replay_indexes=(
            None
            if arrays.indexes is None
            else np.ascontiguousarray(arrays.indexes, dtype=np.int64)
        ),
        action=(
            None
            if actions is None
            else _tensor_from_numpy(torch, actions, pin_memory=pin_memory).to(device=device)
        ),
    )


def _actions_from_arrays(arrays: TrainingArrays) -> np.ndarray | None:
    if arrays.actions is None:
        return None
    actions = np.ascontiguousarray(arrays.actions, dtype=np.int64)
    if actions.shape != (arrays.policies.shape[0],):
        raise ValueError("actions shape must match the policy batch")
    return actions


def _tensor_from_numpy(
    torch_module: Any,
    array: np.ndarray,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    tensor = torch_module.from_numpy(np.ascontiguousarray(array))
    if pin_memory:
        return cast("torch.Tensor", tensor.pin_memory())
    return cast("torch.Tensor", tensor)


def _legal_masks_from_features(features: np.ndarray) -> np.ndarray:
    legal_place = features[:, LEGAL_PLACE_FEATURE_CHANNEL].reshape(-1, BOARD_CELLS) > 0.5
    legal_mask = np.zeros((features.shape[0], BOARD_CELLS + 1), dtype=np.bool_)
    legal_mask[:, :BOARD_CELLS] = legal_place
    legal_mask[:, PASS_ACTION] = True
    return legal_mask


__all__ = ["TrainingArrays", "TrainingBatch", "arrays_to_batch"]
