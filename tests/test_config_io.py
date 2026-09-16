from __future__ import annotations

import json
from pathlib import Path

from great_kingdom_ai.config_io import load_config_object
from great_kingdom_ai.evaluate import load_arena_config


def test_json_loading_remains_available_for_development_tests(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"batch_size": 7}), encoding="utf-8")

    assert load_config_object(config_path, "config")["batch_size"] == 7


def test_remaining_yaml_configs_load_with_existing_values() -> None:
    arena = load_arena_config(
        "configs/runpod/arena.yaml",
        randomize_missing_seed_start=False,
        randomize_missing_gumbel_seed=False,
    )
    fast_matrix = load_arena_config(
        "configs/runpod/fast-matrix.yaml",
        randomize_missing_seed_start=False,
        randomize_missing_gumbel_seed=False,
    )
    full_matrix = load_arena_config(
        "configs/runpod/full-matrix.yaml",
        randomize_missing_seed_start=False,
        randomize_missing_gumbel_seed=False,
    )

    assert (arena.games, arena.batch_size, arena.promotion_threshold) == (60, 16, 0.5)
    assert (fast_matrix.games, fast_matrix.batch_size) == (2, 2)
    assert (full_matrix.games, full_matrix.batch_size) == (20, 16)
