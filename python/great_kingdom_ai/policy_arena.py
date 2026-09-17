"""Batched policy-only arena with model-independent, paired random openings.

Values returned by the shared inference API are ignored. No Q-based policy
improvement, tree search, or Gumbel noise is used to choose a move.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from great_kingdom_ai.arena_types import (
    BLUE,
    ORANGE,
    ArenaConfig,
    ArenaGameResult,
    ArenaReport,
    arena_game_seed,
    other_player,
)
from great_kingdom_ai.self_play_types import MoveLog, SelfPlayState

if TYPE_CHECKING:
    from great_kingdom_ai.arena_runner import ArenaRuntime


def policy_action(logits: Sequence[float], legal_actions: Sequence[int]) -> int:
    """Select the largest legal logit, breaking ties by smallest action index."""
    legal = sorted(legal_actions)
    if not legal:
        raise ValueError("policy arena requires at least one legal action")
    if not all(np.isfinite(logits[action]) for action in legal):
        raise ValueError("policy arena received non-finite legal policy logits")
    return max(legal, key=lambda action: logits[action])


def play_policy_games(
    *,
    states: Sequence[SelfPlayState],
    seeds: Sequence[int],
    candidate_players: Sequence[int],
    candidate_model: Any,
    best_model: Any,
    config: ArenaConfig,
    runtime: ArenaRuntime,
) -> list[ArenaGameResult]:
    """Play one chunk; each game's opening RNG is independent of batch layout."""
    from great_kingdom_ai.arena_runner import evaluate_arena_rows_by_model

    rngs = [random.Random(seed) for seed in seeds]
    moves: list[list[MoveLog]] = [[] for _ in states]
    for turn in range(config.max_turns):
        active = [i for i, state in enumerate(states) if not state.is_terminal()]
        if not active:
            break
        logits: list[list[float]] = []
        if turn >= config.policy_opening_turns:
            logits, _ = evaluate_arena_rows_by_model(
                candidate_model=candidate_model,
                best_model=best_model,
                feature_rows=[states[i].feature_planes() for i in active],
                legal_masks=[states[i].legal_mask() for i in active],
                game_indexes=active,
                current_players=[states[i].current_player() for i in active],
                candidate_players=candidate_players,
                device=config.device,
                runtime=runtime,
            )
        for offset, i in enumerate(active):
            state = states[i]
            legal = sorted(state.legal_actions())
            action = (
                rngs[i].choice(legal)
                if turn < config.policy_opening_turns
                else policy_action(logits[offset], legal)
            )
            moves[i].append(MoveLog(turn=turn, player=state.current_player(), action=action))
            state.apply_action(action)

    results = []
    for i, state in enumerate(states):
        if not state.is_terminal():
            raise RuntimeError(
                f"policy arena exceeded max_turns={config.max_turns} for seed={seeds[i]}"
            )
        winner, end_reason = state.winner(), state.end_reason()
        if winner is None or end_reason is None:
            raise RuntimeError("policy arena stopped before terminal outcome")
        results.append(
            ArenaGameResult(
                seed=seeds[i],
                candidate_player=candidate_players[i],
                best_player=other_player(candidate_players[i]),
                winner=winner,
                end_reason=end_reason,
                moves=moves[i],
                territory_scores=state.territory_scores(),
            )
        )
    return results


def run_policy_arena(
    *,
    candidate_model: Any,
    best_model: Any,
    config: ArenaConfig,
    runtime: ArenaRuntime,
    state_factory: Callable[[], SelfPlayState] | None = None,
    progress_callback: Callable[[int, int, ArenaGameResult], None] | None = None,
) -> ArenaReport:
    from great_kingdom_ai.arena_runner import summarize_arena

    make_state = state_factory if state_factory is not None else runtime.create_game_state
    games = []
    for start in range(0, config.games, config.batch_size):
        indexes = range(start, min(start + config.batch_size, config.games))
        chunk = play_policy_games(
            states=[make_state() for _ in indexes],
            seeds=[arena_game_seed(config, i) for i in indexes],
            candidate_players=[BLUE if i % 2 == 0 else ORANGE for i in indexes],
            candidate_model=candidate_model,
            best_model=best_model,
            config=config,
            runtime=runtime,
        )
        for game in chunk:
            games.append(game)
            if progress_callback is not None:
                progress_callback(len(games), config.games, game)
    return ArenaReport(
        config=config,
        games=games,
        summary=summarize_arena(
            games,
            promotion_threshold=config.promotion_threshold,
            require_side_win_rates_for_promotion=config.require_side_win_rates_for_promotion,
        ),
    )
