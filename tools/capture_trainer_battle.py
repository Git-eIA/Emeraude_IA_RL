"""Throwaway capture tool: spawn the flag-gated route_103 rival and mint trainer_battle.state.

The rival (gObjectEvents template obj[1]: tile (10,3), script Route103_EventScript_Rival) is
HIDDEN by SaveBlock1 hide-flag 0x2D3 (FLAG_HIDE_ROUTE_103_RIVAL). The rival script has no
story-var gate -- the hide-flag is the only gate. (Note: obj[10] gfx 0x40 tile (7,3) flag 0x0382
is Prof. Birch, not the rival; earlier revisions of this tool targeted Birch by mistake.) This
tool clears the rival flag in RAM (cheat-spawn, for the ARTIFACT only), forces a map reload so
the object respawns, navigates to a cell adjacent to (10,3), and spams A through any dialogue
until a trainer battle starts -- then saves states/trainer_battle.state. A bounded retry loop
reloads the state fresh (party restored) and perturbs RNG to absorb wild-encounter attrition on
the northward trip.

The legit path (Birch's post-lab dialog clearing 0x0382) is the deferred scripted campaign
(Option B); this tool only produces the artifact so the two trainer-battle ROM smokes become
load-bearing.

Run in the MAIN repo:
  POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \
    .venv/bin/python tools/capture_trainer_battle.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emulator.buttons as buttons
from emulator.gba import GbaEmulator
from agent.train_fighter import make_move_type_fn
from env.game_state import BattleReader, SAVE_BLOCK1_PTR
from env.grid_navigator import (
    BATTLE_TRANSITION_SETTLE,
    DELTAS,
    RELEASE_FRAMES,
    handle_battle_interruption,
    plan_path_grid,
    probe_step,
    snapshot_settled,
)
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind
from env.world_reader import WorldReader

INPUT_STATE = "states/route_103_reached.state"
OUTPUT_STATE = "states/trainer_battle.state"
FIGHTER_CKPT = "checkpoints/fighter/ppo_fighter_final.zip"

ROUTE_103 = (0, 18)
# FLAG_HIDE_ROUTE_103_RIVAL: gates obj[1] (the real rival, tile (10,3)). The map's
# obj[10] (gfx 0x40, tile (7,3), flag 0x0382) is Prof. Birch, not the rival.
RIVAL_FLAG = 0x2D3
RIVAL_TILE = (10, 3)

_FLAGS_OFF = 0x1270           # SaveBlock1.flags[] offset (Emerald)
_PARTY_COUNT_ADDR = 0x020244E9
_PARTY_ADDR = 0x020244EC
_PARTY_STRIDE = 100
_PARTY_CURHP_OFF = 0x56       # party-mon current HP (u16)
_PARTY_MAXHP_OFF = 0x58       # party-mon max HP (u16)
_STANDABLE = {TileKind.FREE, TileKind.GRASS}
_LOST = ("battle_lost", "battle_timeout", "battle_interrupted")
DELTA_KEYS = {"up": buttons.KEY_UP, "down": buttons.KEY_DOWN,
              "left": buttons.KEY_LEFT, "right": buttons.KEY_RIGHT}

_RELOAD_BUDGET = 30           # bounded map-crossing loop
_NAV_MAX = 250               # bounded nav loop
_TALK_A_PRESSES = 200        # A-spam budget through pre-battle dialogue (rival checkplayergender
                             # + text runs long; the proven probe needed ~200 presses)
_TALK_IDLE_MULT = 3          # idle-frame multiplier between A presses so text advances
_MAX_ATTEMPTS = 5            # retry-RNG budget
_RNG_PERTURB_FRAMES = 17     # idle frames * attempt to vary wild rolls


def _clear_flag(emu, flag_id: int) -> int:
    """Clear a SaveBlock1 flag bit in RAM; return the bit value read back (0 = cleared)."""
    ptr = int.from_bytes(emu.read_bytes(SAVE_BLOCK1_PTR, 4), "little")
    addr = ptr + _FLAGS_OFF + flag_id // 8
    val = emu.read_bytes(addr, 1)[0] & ~(1 << (flag_id % 8))
    emu._core._core.rawWrite8(emu._core._core, addr, -1, val)
    return (emu.read_bytes(addr, 1)[0] >> (flag_id % 8)) & 1


def _heal_party(emu) -> None:
    """Set each party mon's current HP to its max in RAM, so the rival battle is entered at
    full HP. states/route_103_reached.state carries pre-existing attrition (mon at 6/25); the
    artifact must test the Fighter's skill on a fair start, not an unwinnable near-fainted one."""
    count = emu.read_bytes(_PARTY_COUNT_ADDR, 1)[0]
    for i in range(count):
        o = _PARTY_ADDR + i * _PARTY_STRIDE
        maxhp = int.from_bytes(emu.read_bytes(o + _PARTY_MAXHP_OFF, 2), "little")
        emu._core._core.rawWrite8(emu._core._core, o + _PARTY_CURHP_OFF, -1, maxhp & 0xFF)
        emu._core._core.rawWrite8(emu._core._core, o + _PARTY_CURHP_OFF + 1, -1, (maxhp >> 8) & 0xFF)


def _step_until_map(emu, reader, key, *, want_equal=None, want_change_from=None):
    """Hold `key` frame-stepping until map_id reaches/leaves the target. Bounded."""
    last = snapshot_settled(reader)
    for _ in range(_RELOAD_BUDGET):
        emu.step(key, RELEASE_FRAMES)
        emu.step(0, RELEASE_FRAMES)
        here = snapshot_settled(reader)
        if here is None:
            continue
        last = here
        if want_equal is not None and here.map_id == want_equal:
            return here
        if want_equal is None and here.map_id != want_change_from:
            return here
    return last


def _reload_route_103(emu, reader):
    """Leave route_103 south into Oldale then re-enter north; respawns object events."""
    off = _step_until_map(emu, reader, buttons.KEY_DOWN, want_change_from=ROUTE_103)
    print(f"  stepped south -> map={off.map_id if off else None}")
    if off is None or off.map_id == ROUTE_103:
        return None
    back = _step_until_map(emu, reader, buttons.KEY_UP, want_equal=ROUTE_103)
    print(f"  stepped north -> map={back.map_id if back else None} pos={back.pos if back else None}")
    return back if (back is not None and back.map_id == ROUTE_103) else None


def _grass_blocked(snap):
    """Directed edges whose target tile is GRASS (grass allowed but discouraged via cost)."""
    blocked = set()
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            for direction, (dx, dy) in DELTAS.items():
                if snap.classify_at(x + dx, y + dy) is TileKind.GRASS:
                    blocked.add(((x, y), direction))
    return blocked


def _adjacent_targets(snap, tile):
    """Standable cells 4-adjacent to `tile`, each with the heading that faces `tile`."""
    out = []
    for direction, (dx, dy) in DELTAS.items():
        cell = (tile[0] - dx, tile[1] - dy)
        if snap.classify_at(*cell) in _STANDABLE:
            out.append((cell, direction))
    return out


def _pick_stand_cell(reader, here, tile):
    """Shortest grass-avoiding path to a cell adjacent to `tile`. Returns (cell, facing)."""
    snap = GridSnapshot.from_reader(reader.grid_reader, here.map_id)
    grass = _grass_blocked(snap)
    best = None
    for cell, facing in _adjacent_targets(snap, tile):
        path = plan_path_grid(snap, here.pos, cell, blocked=grass)
        if path is not None and (best is None or len(path) < best[2]):
            best = (cell, facing, len(path))
    if best is None:
        for cell, facing in _adjacent_targets(snap, tile):   # grass-allowed fallback
            path = plan_path_grid(snap, here.pos, cell)
            if path is not None and (best is None or len(path) < best[2]):
                best = (cell, facing, len(path))
    return (best[0], best[1]) if best else (None, None)


def _navigate(emu, reader, battle, mtf, predict, stand_cell):
    """Walk to stand_cell, letting the Fighter clear wilds. Returns None on arrival else a
    losing outcome string."""
    presses = 0
    while presses < _NAV_MAX:
        here = snapshot_settled(reader)
        if here is None or here.pos[0] > 255 or here.pos[1] > 255:
            # Transitional RAM frame: pos is garbage; idle a frame and retry.
            emu.step(0, RELEASE_FRAMES)
            presses += 1
            continue
        if battle.battle_starting():
            for _ in range(BATTLE_TRANSITION_SETTLE):
                if reader.in_battle():
                    break
                emu.step(0, RELEASE_FRAMES)
            wild = handle_battle_interruption(emu, reader, mtf, predict)
            print(f"    wild at {here.pos} -> {wild}")
            if wild in _LOST:
                return wild
            continue
        if here.pos == stand_cell:
            return None
        snap = GridSnapshot.from_reader(reader.grid_reader, here.map_id)
        path = plan_path_grid(snap, here.pos, stand_cell)
        if not path:
            print(f"    lost path from {here.pos} to {stand_cell}")
            return "battle_interrupted"
        probe_step(emu, reader, here, path[0])
        presses += 1
    return "battle_timeout"


def _talk_until_battle(emu, reader, battle, facing):
    """Face the rival, then spam A through pre-battle dialogue until a battle starts."""
    # Orient with a few facing presses: the first only turns the player, later ones
    # are absorbed by the solid rival tile. Generous idle gaps let the turn settle.
    facing_key = DELTA_KEYS[facing]
    for _ in range(3):
        emu.step(facing_key, RELEASE_FRAMES)
        emu.step(0, RELEASE_FRAMES * 2)
    for _ in range(_TALK_A_PRESSES):
        if battle.battle_starting():
            return True
        emu.step(buttons.KEY_A, RELEASE_FRAMES)
        emu.step(0, RELEASE_FRAMES * _TALK_IDLE_MULT)
    return battle.battle_starting()


def _scan_objects(emu):
    """Print live gObjectEvents slots (gfx + tile) so the rival spawn can be eyeballed."""
    blob = emu.read_bytes(0x02037350, 0x24 * 16)
    for i in range(16):
        o = i * 0x24
        if not blob[o] & 1:
            continue
        gx = int.from_bytes(blob[o + 0x10:o + 0x12], "little", signed=True)
        gy = int.from_bytes(blob[o + 0x12:o + 0x14], "little", signed=True)
        tag = " <-- RIVAL TILE" if (gx - 7, gy - 7) == RIVAL_TILE else ""
        print(f"    live obj slot={i} gfx=0x{blob[o+5]:02x} tile=({gx-7},{gy-7}){tag}")


def _locate_rival(emu):
    """Return the live map tile (x, y) of the rival object, or None if not loaded.

    The rival's live graphicsId is set at runtime by checkplayergender (May/Brendan),
    so it does not match the ROM template gfx. Locate by proximity to RIVAL_TILE among
    active non-player slots (slot 0 is the player)."""
    blob = emu.read_bytes(0x02037350, 0x24 * 16)
    best = None
    for i in range(1, 16):
        o = i * 0x24
        if not blob[o] & 1:
            continue
        gx = int.from_bytes(blob[o + 0x10:o + 0x12], "little", signed=True)
        gy = int.from_bytes(blob[o + 0x12:o + 0x14], "little", signed=True)
        tile = (gx - 7, gy - 7)
        dist = abs(tile[0] - RIVAL_TILE[0]) + abs(tile[1] - RIVAL_TILE[1])
        if best is None or dist < best[1]:
            best = (tile, dist)
    return best[0] if best is not None else None


def _attempt(emu, state_bytes, mtf, predict, attempt) -> int:
    """Run one full capture attempt from a freshly-reloaded state. Returns 0 on success.

    Each attempt reloads the input state (restoring the party after any whiteout) and
    idles a few RNG-perturbing frames scaled by the attempt index, so wild-encounter
    rolls vary across retries and the northward trip can eventually dodge attrition."""
    emu.load_state(state_bytes)
    emu.step(0, 4 + _RNG_PERTURB_FRAMES * attempt)
    reader = WorldReader(emu.read_bytes)
    battle = BattleReader(emu.read_bytes)

    here = reader.snapshot()
    print(f"start map={here.map_id} pos={here.pos}")
    bit = _clear_flag(emu, RIVAL_FLAG)
    print(f"cleared flag 0x{RIVAL_FLAG:04x} -> bit now {bit}")

    back = _reload_route_103(emu, reader)
    if back is None:
        print("FAILED to reload route_103; aborting")
        return 1

    _scan_objects(emu)
    here = snapshot_settled(reader)
    stand_cell, facing = _pick_stand_cell(reader, here, RIVAL_TILE)
    if stand_cell is None:
        print(f"no path to a cell adjacent to {RIVAL_TILE}; aborting")
        return 1
    print(f"navigate {here.pos} -> stand {stand_cell} face {facing}")
    lost = _navigate(emu, reader, battle, mtf, predict, stand_cell)
    if lost is not None:
        print(f"END nav {lost}")
        return 1

    # Phase 2: retarget to the LIVE rival object position (may differ from ROM template).
    _scan_objects(emu)
    rival = _locate_rival(emu)
    print(f"live rival tile = {rival}")
    if rival is None:
        print("END rival_object_not_loaded")
        return 1
    here = snapshot_settled(reader)
    stand_cell, facing = _pick_stand_cell(reader, here, rival)
    if stand_cell is None:
        print(f"no path to a cell adjacent to live rival {rival}; aborting")
        return 1
    if here.pos != stand_cell:
        print(f"retarget {here.pos} -> stand {stand_cell} face {facing}")
        lost = _navigate(emu, reader, battle, mtf, predict, stand_cell)
        if lost is not None:
            print(f"END retarget nav {lost}")
            return 1

    _heal_party(emu)
    print(f"arrived at {stand_cell}; healed party to full; facing {facing}, spamming A")
    if _talk_until_battle(emu, reader, battle, facing) and battle.is_trainer_battle():
        Path(OUTPUT_STATE).write_bytes(emu.save_state())
        print(f"END rival_confirmed -> saved {OUTPUT_STATE}")
        return 0
    print(f"END no_trainer_battle starting={battle.battle_starting()} "
          f"trainer={battle.is_trainer_battle()} in_battle={reader.in_battle()}")
    return 1


def main() -> int:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    emu = GbaEmulator(rom)
    state_bytes = Path(INPUT_STATE).read_bytes()

    from stable_baselines3 import PPO
    model = PPO.load(FIGHTER_CKPT, device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])
    mtf = make_move_type_fn(emu)

    for attempt in range(_MAX_ATTEMPTS):
        print(f"=== attempt {attempt + 1}/{_MAX_ATTEMPTS} ===")
        if _attempt(emu, state_bytes, mtf, predict, attempt) == 0:
            return 0
    print(f"END exhausted {_MAX_ATTEMPTS} attempts without a trainer battle")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
