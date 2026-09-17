from dataclasses import replace
from types import SimpleNamespace

import great_kingdom_ai.evaluate as arena
import numpy as np
import pytest
from great_kingdom_ai.policy_arena import policy_action


class ToyState:
    def __init__(self):
        self.moves = []

    def is_terminal(self):
        return len(self.moves) == 6

    def current_player(self):
        return 1 + len(self.moves) % 2

    def legal_actions(self):
        return [0, 2, 3]

    def legal_mask(self):
        return [True, False, True, True]

    def feature_planes(self):
        return [float(len(self.moves))]

    def apply_action(self, action):
        assert action in self.legal_actions()
        self.moves.append(action)

    def winner(self):
        return 1 if self.is_terminal() else None

    def end_reason(self):
        return 1 if self.is_terminal() else None

    def territory_scores(self):
        return (3, 1)


@pytest.fixture
def policy_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("policy mode must not create search engines or Rust search batches")

    def evaluate(model, features, masks, **kwargs):
        # The illegal action has the largest logit; value must not affect moves.
        row = [0.0, 100.0, 5.0 if model == "candidate" else 1.0, 2.0]
        return SimpleNamespace(
            policy_logits=np.array([row] * len(features)), value=np.full(len(features), np.nan)
        )

    monkeypatch.setattr(arena, "create_core_game_state", ToyState)
    monkeypatch.setattr(arena, "create_core_search_engine", forbidden)
    monkeypatch.setattr(arena, "create_core_arena_batch", forbidden)
    monkeypatch.setattr(arena, "evaluate_feature_batch_logits_values", evaluate)


def test_policy_legal_argmax_and_ties():
    assert policy_action([2.0, 100.0, 2.0], [2, 0]) == 0
    with pytest.raises(ValueError, match="non-finite"):
        policy_action([np.nan], [0])


def test_policy_pairing_batch_invariance_and_model_routing(policy_runtime):
    config = arena.ArenaConfig(
        games=6,
        batch_size=3,
        paired_seeds=True,
        seed_start=1000,
        action_selection="policy",
        policy_opening_turns=2,
        gumbel_simulations=0,
        max_turns=6,
    )
    reports = [
        arena.run_arena(
            candidate_model="candidate", best_model="best", config=replace(config, batch_size=size)
        )
        for size in [1, 3, 6]
    ]
    assert reports[0].games == reports[1].games == reports[2].games
    games = reports[0].games
    for i in range(0, 6, 2):
        assert games[i].moves[:2] == games[i + 1].moves[:2]
        for game in games[i : i + 2]:
            for move in game.moves[2:]:
                assert move.action == (2 if move.player == game.candidate_player else 3)
    assert len({tuple(m.action for m in g.moves[:2]) for g in games}) > 1
    assert reports[0].to_dict()["config"]["action_selection"] == "policy"


def test_policy_single_game_and_batch_entrypoints(policy_runtime):
    config = arena.ArenaConfig(
        games=2, action_selection="policy", policy_opening_turns=0, max_turns=6
    )
    game = arena.play_arena_game(
        seed=0, candidate_model="candidate", best_model="best", candidate_player=1, config=config
    )
    assert [m.action for m in game.moves] == [2, 3, 2, 3, 2, 3]
    report = arena.run_arena_batched(candidate_model="candidate", best_model="best", config=config)
    assert len(report.games) == 2
    with pytest.raises(RuntimeError, match="max_turns"):
        arena.run_arena(
            candidate_model="candidate", best_model="best", config=replace(config, max_turns=5)
        )


def test_policy_cli_and_onnx_rejection(policy_runtime):
    args = arena.build_parser().parse_args(
        [
            "--candidate",
            "a.pt",
            "--best",
            "b.pt",
            "--report",
            "out.json",
            "--action-selection",
            "policy",
            "--gumbel-simulations",
            "0",
            "--policy-opening-turns",
            "2",
        ]
    )
    config = arena._config_from_args(args)
    arena.validate_arena_config(config)
    assert config.action_selection == "policy"
    with pytest.raises(ValueError, match="backend=pytorch"):
        arena.run_arena_checkpoints_onnx(
            candidate_checkpoint="missing.pt", best_checkpoint="missing.pt", config=config
        )
    with pytest.raises(ValueError, match="gumbel_simulations"):
        arena.validate_arena_config(replace(config, action_selection="gumbel"))
