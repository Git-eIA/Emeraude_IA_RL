"""Shared battle-turn choreography: one turn of input + the 17-dim observation.

Extracted from BattleEmeraldEnv so the training env and the live Fighter player
share exactly one copy of the turn timing and the observation layout.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from emulator import buttons
from env.game_state import BattleState

# Button pulse: hold then release, so each press is a distinct edge. Pressing
# with no release frame in between makes the GBA debounce consecutive holds
# into a single press, which stalls all menu navigation.
PRESS_HOLD_FRAMES = 6
PRESS_RELEASE_FRAMES = 10
SETTLE_FRAMES = 8  # let a freshly-opened menu accept input before pressing
OPEN_MENU_TRIES = 10  # A-presses allowed to open the move list from the menu
MAX_ADVANCE_PRESSES = 120  # cap the A-spam loop so a stuck battle can't hang
NUM_TYPES = 18  # types 0..17
MAX_PP = 40.0
OBS_SIZE = 17

MoveTypeFn = Callable[[int], int]


def press(emulator: Any, key: int) -> None:
    """Hold then release so the GBA registers a distinct press."""
    emulator.step(key, PRESS_HOLD_FRAMES)
    emulator.step(0, PRESS_RELEASE_FRAMES)


def select_move(emulator: Any, reader: Any, action: int) -> None:
    """From the action menu: open the move list, move to `action`, commit.

    Pressing A until the menu opens absorbs an input eaten by the open animation.
    """
    for _ in range(OPEN_MENU_TRIES):
        if not reader.at_action_menu():
            break
        press(emulator, buttons.KEY_A)
    for _ in range(action):
        press(emulator, buttons.KEY_DOWN)
    press(emulator, buttons.KEY_A)


def advance_to_menu(emulator: Any, reader: Any) -> None:
    """Press A to clear dialogue until back at the action menu or battle end."""
    for _ in range(MAX_ADVANCE_PRESSES):
        state = reader.battle_state()
        if state.outcome != 0 or not state.in_battle:
            return
        if reader.at_action_menu():
            emulator.step(0, SETTLE_FRAMES)
            return
        press(emulator, buttons.KEY_A)


def observation(state: BattleState, move_type_fn: MoveTypeFn) -> np.ndarray:
    """The 17-dim RAM observation the Fighter policy was trained on."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[0] = _frac(state.my_hp, state.my_max_hp)
    obs[1] = min(state.my_level / 100.0, 1.0)
    obs[2] = _frac(state.opp_hp, state.opp_max_hp)
    obs[3] = min(state.opp_level / 100.0, 1.0)
    # Clamp type encodings: a RAM-transition garbage read can return a byte
    # above the valid 0..17 range, which would break the Box [0, 1] bound.
    obs[4] = min(state.my_types[0] / NUM_TYPES, 1.0)
    obs[5] = min(state.my_types[1] / NUM_TYPES, 1.0)
    obs[6] = min(state.opp_types[0] / NUM_TYPES, 1.0)
    obs[7] = min(state.opp_types[1] / NUM_TYPES, 1.0)
    for i, move in enumerate(state.my_moves):
        obs[8 + 2 * i] = min(move_type_fn(move.move_id) / NUM_TYPES, 1.0)
        obs[9 + 2 * i] = min(move.pp / MAX_PP, 1.0)
    obs[16] = 1.0 if state.in_battle else 0.0
    return obs


def _frac(hp: int, max_hp: int) -> float:
    return hp / max_hp if max_hp > 0 else 0.0
