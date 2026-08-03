"""Drive a live, ongoing battle with a trained Fighter policy to an outcome.

Unlike BattleEmeraldEnv (which trains, resets, and shapes reward), this just
plays: the battle is already running (grind triggered it), and a caller-injected
`predict` chooses each move. Kept free of SB3/torch — production wraps a loaded
PPO model into `predict`.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from env.battle_turn import advance_to_menu, observation, select_move
from env.game_state import BattleReader

PredictFn = Callable[[np.ndarray], int]
MoveTypeFn = Callable[[int], int]


def play_battle(
    emulator: Any,
    move_type_fn: MoveTypeFn,
    predict: PredictFn,
    max_turns: int = 64,
) -> str:
    """Play the ongoing battle to the end.

    Returns "won" (outcome bit 0x1 set), "lost" (any other terminal outcome),
    or "battle_timeout" (max_turns reached without a terminal outcome).
    """
    reader = BattleReader(emulator.read_bytes)
    advance_to_menu(emulator, reader)  # reach the first action menu (or battle end)
    for _ in range(max_turns):
        state = reader.battle_state()
        if state.outcome != 0 or not state.in_battle:
            return _result(state.outcome)
        action = predict(observation(state, move_type_fn))
        select_move(emulator, reader, int(action))
        advance_to_menu(emulator, reader)
    state = reader.battle_state()
    if state.outcome != 0 or not state.in_battle:
        return _result(state.outcome)
    return "battle_timeout"


def play_trainer_battle(
    emulator: Any,
    move_type_fn: MoveTypeFn,
    predict: PredictFn,
    max_turns: int = 128,
) -> str:
    """Play an ongoing multi-Pokémon trainer battle to the end.

    Unlike play_battle (wild, single opponent), a trainer sends out the next
    Pokémon when its active one faints; opp max_hp reads 0 for a tick during
    send-out, so this stops ONLY on a terminal outcome, never on not in_battle.

    Returns "won" (outcome bit 0x1 set), "lost" (any other terminal outcome),
    or "battle_timeout" (max_turns reached without a terminal outcome).
    """
    reader = BattleReader(emulator.read_bytes)
    advance_to_menu(emulator, reader, wait_through_faint=True)
    for _ in range(max_turns):
        state = reader.battle_state()
        if state.outcome != 0:
            return _result(state.outcome)
        action = predict(observation(state, move_type_fn))
        select_move(emulator, reader, int(action))
        advance_to_menu(emulator, reader, wait_through_faint=True)
    state = reader.battle_state()
    if state.outcome != 0:
        return _result(state.outcome)
    return "battle_timeout"


def _result(outcome: int) -> str:
    # NOTE: assumes predict never escapes; a wild grind battle only ends by a
    # terminal outcome, so battle-end with outcome==0 is not reachable here.
    return "won" if outcome & 0x1 else "lost"
