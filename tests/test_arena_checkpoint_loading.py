"""Checkpoint compatibility and real mixed-model matrix integration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from great_kingdom_ai.evaluate import load_model_from_checkpoint  # noqa: E402
from great_kingdom_ai.klent.checkpoint import KlentTrainState, save_klent_checkpoint  # noqa: E402
from great_kingdom_ai.klent.targets import (  # noqa: E402
    legal_mask_from_features,
    masked_state_value,
)
from great_kingdom_ai.klent.types import KlentConfig  # noqa: E402
from great_kingdom_ai.model import ModelConfig, PolicyValueNetwork  # noqa: E402


def save_model(path: Path, *, klent: bool) -> PolicyValueNetwork:
    model = PolicyValueNetwork(
        ModelConfig(
            channels=4,
            residual_blocks=0,
            value_hidden=4,
            action_value_head=klent,
        )
    )
    with torch.no_grad():
        model.policy_pass[-1].weight.zero_()
        model.policy_pass[-1].bias.fill_(100)
    if klent:
        save_klent_checkpoint(
            KlentTrainState(
                model=model,
                optimizer=torch.optim.AdamW(model.parameters()),
                iteration=2,
                total_steps=3,
                klent_config=KlentConfig(),
                model_preset="custom",
            ),
            path,
        )
    else:
        torch.save({"model_config": asdict(model.config), "model_state": model.state_dict()}, path)
    return model.eval()


def test_klent_loader_preserves_masked_q_value(tmp_path: Path) -> None:
    core = pytest.importorskip("great_kingdom_core")
    path = tmp_path / "klent.pt"
    source = save_model(path, klent=True)
    loaded = load_model_from_checkpoint(path)
    features = torch.tensor(core.GameState().feature_planes()).reshape(1, 11, 9, 9)
    with torch.inference_mode():
        logits, q = source.forward_q(features)
        actual_logits, actual_value = loaded(features)
    torch.testing.assert_close(actual_logits, logits)
    torch.testing.assert_close(
        actual_value,
        masked_state_value(
            logits,
            q,
            legal_mask_from_features(features),
        ),
    )
    assert not loaded.training


@pytest.mark.parametrize("prefer_ema", [False, True])
def test_legacy_loader_preserves_ema_choice(tmp_path: Path, prefer_ema: bool) -> None:
    path = tmp_path / "legacy.pt"
    model = save_model(path, klent=False)
    checkpoint = torch.load(path, weights_only=False)
    ema = {key: value.clone() for key, value in model.state_dict().items()}
    ema["policy_pass.2.bias"].fill_(200)
    checkpoint["ema_model_state"] = ema
    torch.save(checkpoint, path)
    loaded = load_model_from_checkpoint(path, prefer_ema=prefer_ema)
    assert loaded.policy_pass[-1].bias.item() == (200 if prefer_ema else 100)


def test_invalid_klent_and_missing_weights_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    save_model(path, klent=False)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["algorithm"] = "klent"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="action_value_head"):
        load_model_from_checkpoint(path)
    save_model(path, klent=True)
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["model_state"]["policy_pass.2.bias"]
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model_from_checkpoint(path)


def test_pt_folder_matrix_runs_mixed_and_klent_pairs(tmp_path: Path) -> None:
    pytest.importorskip("great_kingdom_core")
    for name, klent in [("legacy", False), ("klent-a", True), ("klent-b", True)]:
        save_model(tmp_path / f"{name}.pt", klent=klent)
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_pt_folder_matrix_arena.py"),
            str(tmp_path),
            "--arena-config",
            str(root / "configs/runpod/fast-matrix.yaml"),
            "--device",
            "cpu",
            "--games",
            "2",
            "--gumbel-simulations",
            "2",
            "--gumbel-max-considered-actions",
            "2",
        ],
        cwd=root,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": str(root / "python"),
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    summary = json.loads((tmp_path / "matrix-arena-reports/summary.json").read_text())
    assert summary["pair_count"] == 3
    assert len(summary["ranking"]) == 3
    assert all(match["games"] == 2 for match in summary["matches"])
    assert all(match["candidate_blue_games"] == 1 for match in summary["matches"])
    assert all(match["candidate_orange_games"] == 1 for match in summary["matches"])
