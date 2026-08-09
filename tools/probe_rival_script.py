"""Throwaway probe: decode route_103 rival trigger gate from ROM event scripts.

Clears FLAG_HIDE_ROUTE_103_RIVAL (0x2D3) — the CORRECT rival flag, not 0x0382
which is FLAG_HIDE_ROUTE_103_BIRCH — reloads the map, dumps MapEvents counts +
all event arrays (ObjectEventTemplates, CoordEvents, BgEvents), then dumps ~128
bytes of ROM bytecode at each rival object's script ptr and any coord/bg event
script ptrs near the rival tile. Then attempts to arm and trigger the battle.

Run:
    cd /Users/_eloi/Projets/Emu-scripted-rival-capture
    POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \
        .venv/bin/python tools/probe_rival_script.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import emulator.buttons as buttons
from emulator.gba import GbaEmulator
from env.game_state import BattleReader, SAVE_BLOCK1_PTR
from env.grid_navigator import RELEASE_FRAMES, plan_path_grid, snapshot_settled
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import MapGridReader
from env.world_reader import WorldReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_STATE = "states/route_103_reached.state"
OUTPUT_STATE = "states/trainer_battle.state"
ROUTE_103 = (0, 18)

# Correct rival hide flag (pokeemerald flags.h: FLAG_HIDE_ROUTE_103_RIVAL 0x2D3).
# Prior work used 0x0382 = FLAG_HIDE_ROUTE_103_BIRCH — wrong object.
RIVAL_FLAG = 0x02D3

# SaveBlock1 offsets (verified in env/game_state.py against pokeemerald source).
_FLAGS_OFF = 0x1270   # u8 flags[]
_VARS_OFF = 0x139C    # u16 vars[]
_VARS_START = 0x4000  # first var id

# RAM / ROM address spaces
_EWRAM_S = 0x02000000
_EWRAM_E = 0x02040000
_ROM_S = 0x08000000
_ROM_E = 0x0C000000

# gMapHeader (BPEF == BPEE layout, verified by probe_rival_flag.py)
_G_MAP_HEADER = 0x02037318
_EVENTS_OFF = 0x04  # MapHeader.events -> ptr to struct MapEvents

# struct MapEvents offsets (pokeemerald include/global.fieldmap.h)
_ME_OBJ_COUNT = 0x00    # u8
_ME_WARP_COUNT = 0x01   # u8
_ME_COORD_COUNT = 0x02  # u8
_ME_BG_COUNT = 0x03     # u8
_ME_OBJ_PTR = 0x04      # ptr to ObjectEventTemplate[]
_ME_WARP_PTR = 0x08     # ptr to WarpEvent[]
_ME_COORD_PTR = 0x0C    # ptr to CoordEvent[]
_ME_BG_PTR = 0x10       # ptr to BgEvent[]

# struct ObjectEventTemplate (stride 0x18, from global.fieldmap.h)
_OBJ_STRIDE = 0x18
_OT_LOCAL_ID = 0x00    # u8
_OT_GFX = 0x01        # u8 graphicsId
_OT_X = 0x04          # s16
_OT_Y = 0x06          # s16
_OT_TRAINER_TYPE = 0x0C  # u16 trainerType  (NOTE: +0x0E is trainerRange!)
_OT_SCRIPT = 0x10     # const u8 * script ptr
_OT_FLAG = 0x14       # u16 flagId

# struct CoordEvent (stride 0x10, from global.fieldmap.h)
# s16 x @0x00, s16 y @0x02, u8 elevation @0x04, u16 trigger(var) @0x06,
# u16 index(value) @0x08, const u8 *script @0x0C
_CE_STRIDE = 0x10
_CE_X = 0x00
_CE_Y = 0x02
_CE_VAR = 0x06
_CE_VAL = 0x08
_CE_SCRIPT = 0x0C

# struct BgEvent (stride 0x0C)
# u16 x @0x00, u16 y @0x02, u8 elevation @0x04, u8 kind @0x05,
# script/union @0x08
_BE_STRIDE = 0x0C
_BE_X = 0x00
_BE_Y = 0x02
_BE_SCRIPT = 0x08

# Rival graphicsId values.
# In BPEF (French): gfx=0xf0 is the rival object (flag=0x2D3=FLAG_HIDE_ROUTE_103_RIVAL).
# gfx=0x40 is Birch on route_103 (flag=0x0382=FLAG_HIDE_ROUTE_103_BIRCH) — NOT the rival.
# gfx=0x08/0x09 are the BPEE names kept for safety.
_RIVAL_GFX = {0x08, 0x09, 0xf0}  # 0x40 = Birch, NOT rival in BPEF

# Tile proximity threshold for coord/bg event filtering
_NEARBY = 5

# Nav budget constants
_RELOAD_BUDGET = 60
_TALK_BUDGET = 50
_SCRIPT_DUMP_BYTES = 128


# ---------------------------------------------------------------------------
# Low-level RAM helpers
# ---------------------------------------------------------------------------

def _u8(emu, addr: int) -> int:
    return emu.read_bytes(addr, 1)[0]


def _u16(emu, addr: int) -> int:
    return int.from_bytes(emu.read_bytes(addr, 2), "little")


def _s16(emu, addr: int) -> int:
    return int.from_bytes(emu.read_bytes(addr, 2), "little", signed=True)


def _u32(emu, addr: int) -> int:
    return int.from_bytes(emu.read_bytes(addr, 4), "little")


def _rom_bytes(emu, rom_ptr: int, n: int) -> bytes:
    """Read n bytes from a GBA ROM pointer (strip the 0x08000000 bank bit)."""
    if not _ROM_S <= rom_ptr < _ROM_E:
        return b""
    return emu.read_bytes(rom_ptr, n)


def _sb1_ptr(emu) -> int | None:
    ptr = _u32(emu, SAVE_BLOCK1_PTR)
    return ptr if _EWRAM_S <= ptr < _EWRAM_E else None


def _flag_set(emu, flag_id: int) -> bool | None:
    """Read a SaveBlock1 flag bit; None if ptr invalid."""
    ptr = _sb1_ptr(emu)
    if ptr is None or flag_id == 0:
        return None
    byte = _u8(emu, ptr + _FLAGS_OFF + flag_id // 8)
    return bool(byte & (1 << (flag_id % 8)))


def _read_var(emu, var_id: int) -> int | None:
    """Read a SaveBlock1 event var by var_id (e.g. 0x4000+offset)."""
    ptr = _sb1_ptr(emu)
    if ptr is None or var_id < _VARS_START:
        return None
    idx = var_id - _VARS_START
    return _u16(emu, ptr + _VARS_OFF + idx * 2)


# ---------------------------------------------------------------------------
# Flag manipulation
# ---------------------------------------------------------------------------

def _clear_flag(emu, flag_id: int) -> int:
    """Clear a SaveBlock1 flag bit in RAM; return bit read back (0 = cleared)."""
    ptr = _sb1_ptr(emu)
    if ptr is None:
        return -1
    addr = ptr + _FLAGS_OFF + flag_id // 8
    val = _u8(emu, addr) & ~(1 << (flag_id % 8))
    emu._core._core.rawWrite8(emu._core._core, addr, -1, val)
    return (_u8(emu, addr) >> (flag_id % 8)) & 1


def _set_var(emu, var_id: int, value: int) -> None:
    """Write a u16 value into a SaveBlock1 event var."""
    ptr = _sb1_ptr(emu)
    if ptr is None:
        return
    addr = ptr + _VARS_OFF + (var_id - _VARS_START) * 2
    lo = value & 0xFF
    hi = (value >> 8) & 0xFF
    emu._core._core.rawWrite8(emu._core._core, addr, -1, lo)
    emu._core._core.rawWrite8(emu._core._core, addr + 1, -1, hi)


# ---------------------------------------------------------------------------
# Map reload
# ---------------------------------------------------------------------------

def _reload_route_103(emu, reader) -> bool:
    """Walk south off route_103 into Oldale then re-enter north; respawns objects."""
    # Step south until off ROUTE_103
    for _ in range(_RELOAD_BUDGET):
        emu.step(buttons.KEY_DOWN, RELEASE_FRAMES)
        ps = reader.snapshot()
        if ps is not None and ps.map_id != ROUTE_103:
            print(f"  left route_103 -> map={ps.map_id}")
            break
    else:
        print("  ERROR: still on route_103 after south walk")
        return False

    # Step north until back on ROUTE_103
    for _ in range(_RELOAD_BUDGET):
        emu.step(buttons.KEY_UP, RELEASE_FRAMES)
        ps = reader.snapshot()
        if ps is not None and ps.map_id == ROUTE_103:
            print(f"  re-entered route_103 at pos={ps.pos}")
            return True

    print("  ERROR: could not re-enter route_103")
    return False


# ---------------------------------------------------------------------------
# Event array dump helpers
# ---------------------------------------------------------------------------

def _dump_object_events(emu, events_ptr: int) -> list[dict]:
    """Dump all ObjectEventTemplates; return list of dicts."""
    count = _u8(emu, events_ptr + _ME_OBJ_COUNT)
    obj_ptr = _u32(emu, events_ptr + _ME_OBJ_PTR)
    print(f"\n--- ObjectEvents count={count} base={hex(obj_ptr)} ---")
    result = []
    if not _ROM_S <= obj_ptr < _ROM_E:
        print("  objectEvents ptr not in ROM; skipping")
        return result
    for i in range(count):
        base = obj_ptr + i * _OBJ_STRIDE
        local_id = _u8(emu, base + _OT_LOCAL_ID)
        gfx = _u8(emu, base + _OT_GFX)
        x = _s16(emu, base + _OT_X)
        y = _s16(emu, base + _OT_Y)
        trainer_type = _u16(emu, base + _OT_TRAINER_TYPE)
        trainer_range = _u16(emu, base + _OT_TRAINER_TYPE + 2)  # +0x0E
        script_ptr = _u32(emu, base + _OT_SCRIPT)
        flag_id = _u16(emu, base + _OT_FLAG)
        flag_val = _flag_set(emu, flag_id)
        hidden = "HIDDEN" if flag_val else ("visible" if flag_val is False else "always")
        rival_tag = " <-- RIVAL" if gfx in _RIVAL_GFX else ""
        trainer_tag = f" <-- TRAINER(type={trainer_type})" if trainer_type else ""
        print(
            f"  obj[{i}] localId={local_id} gfx=0x{gfx:02x} tile=({x},{y}) "
            f"trainerType={trainer_type} trainerRange={trainer_range} "
            f"flagId=0x{flag_id:04x}({hidden}) script={hex(script_ptr)}"
            f"{rival_tag}{trainer_tag}"
        )
        entry = dict(
            idx=i, local_id=local_id, gfx=gfx, x=x, y=y,
            trainer_type=trainer_type, trainer_range=trainer_range,
            script_ptr=script_ptr, flag_id=flag_id, flag_val=flag_val,
        )
        result.append(entry)
    return result


def _dump_coord_events(emu, events_ptr: int, near_x: int, near_y: int) -> list[dict]:
    """Dump CoordEvents; highlight those near (near_x, near_y)."""
    count = _u8(emu, events_ptr + _ME_COORD_COUNT)
    ptr = _u32(emu, events_ptr + _ME_COORD_PTR)
    print(f"\n--- CoordEvents count={count} base={hex(ptr)} ---")
    result = []
    if count == 0 or not _ROM_S <= ptr < _ROM_E:
        print("  (none or ptr out of ROM)")
        return result
    for i in range(count):
        base = ptr + i * _CE_STRIDE
        x = _s16(emu, base + _CE_X)
        y = _s16(emu, base + _CE_Y)
        var_id = _u16(emu, base + _CE_VAR)
        val = _u16(emu, base + _CE_VAL)
        script_ptr = _u32(emu, base + _CE_SCRIPT)
        nearby = abs(x - near_x) <= _NEARBY and abs(y - near_y) <= _NEARBY
        tag = " <-- NEAR RIVAL" if nearby else ""
        print(
            f"  coord[{i}] ({x},{y}) var=0x{var_id:04x} val={val} "
            f"script={hex(script_ptr)}{tag}"
        )
        result.append(dict(idx=i, x=x, y=y, var_id=var_id, val=val,
                           script_ptr=script_ptr, nearby=nearby))
    return result


def _dump_bg_events(emu, events_ptr: int, near_x: int, near_y: int) -> list[dict]:
    """Dump BgEvents; highlight those near (near_x, near_y)."""
    count = _u8(emu, events_ptr + _ME_BG_COUNT)
    ptr = _u32(emu, events_ptr + _ME_BG_PTR)
    print(f"\n--- BgEvents count={count} base={hex(ptr)} ---")
    result = []
    if count == 0 or not _ROM_S <= ptr < _ROM_E:
        print("  (none or ptr out of ROM)")
        return result
    for i in range(count):
        base = ptr + i * _BE_STRIDE
        x = _u16(emu, base + _BE_X)
        y = _u16(emu, base + _BE_Y)
        kind = _u8(emu, base + 0x05)
        script_ptr = _u32(emu, base + _BE_SCRIPT)
        nearby = abs(x - near_x) <= _NEARBY and abs(y - near_y) <= _NEARBY
        tag = " <-- NEAR RIVAL" if nearby else ""
        print(f"  bg[{i}] ({x},{y}) kind={kind} script={hex(script_ptr)}{tag}")
        result.append(dict(idx=i, x=x, y=y, kind=kind,
                           script_ptr=script_ptr, nearby=nearby))
    return result


def _dump_script_bytes(emu, label: str, script_ptr: int) -> None:
    """Hex-dump _SCRIPT_DUMP_BYTES ROM bytes at script_ptr."""
    if not _ROM_S <= script_ptr < _ROM_E:
        print(f"  {label}: ptr {hex(script_ptr)} not in ROM")
        return
    raw = _rom_bytes(emu, script_ptr, _SCRIPT_DUMP_BYTES)
    hex_rows = []
    for off in range(0, len(raw), 16):
        chunk = raw[off:off + 16]
        h = " ".join(f"{b:02x}" for b in chunk)
        hex_rows.append(f"    {hex(script_ptr + off)}: {h}")
    print(f"  {label} script @ {hex(script_ptr)}:")
    print("\n".join(hex_rows))


# ---------------------------------------------------------------------------
# Nav + battle-trigger
# ---------------------------------------------------------------------------

_DIR_KEY = {"up": buttons.KEY_UP, "down": buttons.KEY_DOWN,
             "left": buttons.KEY_LEFT, "right": buttons.KEY_RIGHT}


def _grass_free_snap(snap: GridSnapshot) -> GridSnapshot:
    """Return a copy of snap with GRASS replaced by WALL so A* avoids wild encounters."""
    from env.map_grid_reader import TileKind
    new_tiles = tuple(
        tuple(TileKind.WALL if cell is TileKind.GRASS else cell for cell in row)
        for row in snap.tiles
    )
    return GridSnapshot(map_id=snap.map_id, width=snap.width,
                        height=snap.height, tiles=new_tiles)


def _walk_adjacent(emu, reader, target_x: int, target_y: int) -> bool:
    """Walk to the cell one tile south of (target_x, target_y) using A* on the
    live grid, then face UP toward the rival. Falls back to trying all four
    cardinal neighbours if south is blocked."""
    gr = MapGridReader(emu.read_bytes)
    ps0 = snapshot_settled(reader)
    snap = GridSnapshot.from_reader(gr, ps0.map_id if ps0 else (0, 18))
    if snap is None:
        print("  GridSnapshot.from_reader returned None; aborting nav")
        return False

    if ps0 is None:
        return False
    start = ps0.pos

    # Try standing one tile south first, then try other neighbours.
    candidates = [
        (target_x, target_y + 1),
        (target_x, target_y - 1),
        (target_x - 1, target_y),
        (target_x + 1, target_y),
    ]
    path: list[str] | None = None
    stand_cell = None
    no_grass = _grass_free_snap(snap)
    for cell in candidates:
        p = plan_path_grid(no_grass, start, cell)
        if p is not None:
            path = p
            stand_cell = cell
            break

    if path is None or stand_cell is None:
        print(f"  A* found no path from {start} to any cell adjacent to ({target_x},{target_y})")
        return False

    print(f"  A* path: {start} -> {stand_cell} ({len(path)} steps)")

    # Execute step-by-step; re-plan every step so NPC/tile drift doesn't derail us.
    # If a wild battle starts mid-nav, run away (B + up = run in Emerald).
    _MAX_STEPS = 400
    battle_tmp = BattleReader(emu.read_bytes)
    for step_i in range(_MAX_STEPS):
        # Should not reach here if grass is excluded from the path.
        if battle_tmp.battle_starting():
            print(f"  unexpected wild battle at step {step_i}; aborting nav")
            break

        ps_cur = snapshot_settled(reader)
        if ps_cur is None:
            continue
        cur = ps_cur.pos
        if cur == stand_cell:
            print(f"  arrived at {stand_cell} after {step_i} steps")
            return True
        # Re-plan from current position each step (grass excluded).
        snap2 = GridSnapshot.from_reader(MapGridReader(emu.read_bytes), ps_cur.map_id)
        if snap2 is None:
            continue
        next_path = plan_path_grid(_grass_free_snap(snap2), cur, stand_cell)
        if not next_path:
            print(f"  A* lost path at {cur} after {step_i} steps")
            break
        key = _DIR_KEY.get(next_path[0])
        if key is not None:
            emu.step(key, RELEASE_FRAMES)

    ps_final = snapshot_settled(reader)
    arrived = ps_final is not None and ps_final.pos == stand_cell
    print(f"  arrived={arrived} pos={ps_final.pos if ps_final else '?'} want={stand_cell}")
    return arrived


def _scan_live_objects(emu) -> list[tuple[int, int, int]]:
    """Return list of (gfx, x, y) for each active live object in EWRAM."""
    # gObjectEvents[] lives at 0x02037350 in BPEF; stride 0x24; max 16 entries.
    # Slot active flag: byte 0 bit 0; graphicsId: byte 5; map_x: u16 @ 0x10; map_y: u16 @ 0x12.
    _LIVE_OBJ_BASE = 0x02037350
    _LIVE_OBJ_STRIDE = 0x24
    _LIVE_OBJ_MAX = 16
    blob = emu.read_bytes(_LIVE_OBJ_BASE, _LIVE_OBJ_STRIDE * _LIVE_OBJ_MAX)
    result = []
    for i in range(_LIVE_OBJ_MAX):
        o = i * _LIVE_OBJ_STRIDE
        if not (blob[o] & 1):   # inactive slot
            continue
        gfx = blob[o + 5]
        x = int.from_bytes(blob[o + 0x10: o + 0x12], "little") - 7
        y = int.from_bytes(blob[o + 0x12: o + 0x14], "little") - 7
        result.append((gfx, x, y))
    return result


def _face_and_talk(emu, reader, battle: BattleReader,
                   facing_key: int) -> bool:
    """Face the rival (press facing_key once to orient) then spam A."""
    # Dump live objects to confirm rival is spawned.
    live = _scan_live_objects(emu)
    print(f"  live objects ({len(live)}): "
          + ", ".join(f"gfx=0x{g:02x}@({x},{y})" for g, x, y in live))

    # Orient: press facing_key 3 times to ensure the player faces the rival.
    # (First press may only turn; subsequent presses are absorbed by the NPC tile.)
    for _ in range(3):
        emu.step(facing_key, RELEASE_FRAMES)
        emu.step(0, RELEASE_FRAMES * 2)

    # Spam A with generous idle gaps until battle starts or budget exhausted.
    for _ in range(_TALK_BUDGET * 4):
        if battle.battle_starting():
            return True
        emu.step(buttons.KEY_A, RELEASE_FRAMES)
        emu.step(0, RELEASE_FRAMES * 3)
    return battle.battle_starting()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:  # noqa: C901 (complexity fine for a throwaway probe)
    rom = os.environ.get("POKEMON_EMERALD_ROM", "")
    if not rom:
        print("ERROR: POKEMON_EMERALD_ROM not set")
        return 1

    emu = GbaEmulator(rom)
    state_bytes = Path(INPUT_STATE).read_bytes()
    emu.load_state(state_bytes)
    emu.step(0, 4)

    reader = WorldReader(emu.read_bytes)
    battle = BattleReader(emu.read_bytes)
    ps = reader.snapshot()
    print(f"=== PROBE START  map={ps.map_id} pos={ps.pos} ===\n")

    # ------------------------------------------------------------------
    # Step 1: Clear RIVAL flag (correct: 0x2D3 = FLAG_HIDE_ROUTE_103_RIVAL)
    # ------------------------------------------------------------------
    print(f"[1] FLAG_HIDE_ROUTE_103_RIVAL = 0x{RIVAL_FLAG:04x}")
    before = _flag_set(emu, RIVAL_FLAG)
    bit = _clear_flag(emu, RIVAL_FLAG)
    print(f"    flag before={before} -> after bit={bit} (0=cleared)")

    # Also read key story vars for context
    var_birch = _read_var(emu, 0x4085)   # VAR_BIRCH_LAB_STATE (common idx)
    var_starter = _read_var(emu, 0x4054)  # VAR_STARTER_MON
    print(f"    VAR_BIRCH_LAB_STATE(0x4085)={var_birch}  "
          f"VAR_STARTER_MON(0x4054)={var_starter}")

    # ------------------------------------------------------------------
    # Step 2: Reload route_103 so object events respawn
    # ------------------------------------------------------------------
    print("\n[2] Reloading route_103 ...")
    ok = _reload_route_103(emu, reader)
    if not ok:
        print("FATAL: map reload failed")
        return 1
    emu.step(0, RELEASE_FRAMES * 4)

    # ------------------------------------------------------------------
    # Step 3: Dump MapEvents
    # ------------------------------------------------------------------
    print("\n[3] Dumping MapEvents ...")
    events_ptr = _u32(emu, _G_MAP_HEADER + _EVENTS_OFF)
    print(f"    gMapHeader @ {hex(_G_MAP_HEADER)}  .events = {hex(events_ptr)}")
    if not _ROM_S <= events_ptr < _ROM_E:
        print("    ERROR: events ptr not in ROM")
        return 1

    obj_count = _u8(emu, events_ptr + _ME_OBJ_COUNT)
    warp_count = _u8(emu, events_ptr + _ME_WARP_COUNT)
    coord_count = _u8(emu, events_ptr + _ME_COORD_COUNT)
    bg_count = _u8(emu, events_ptr + _ME_BG_COUNT)
    obj_ptr = _u32(emu, events_ptr + _ME_OBJ_PTR)
    coord_ptr = _u32(emu, events_ptr + _ME_COORD_PTR)
    bg_ptr = _u32(emu, events_ptr + _ME_BG_PTR)
    print(
        f"    counts: obj={obj_count} warp={warp_count} "
        f"coord={coord_count} bg={bg_count}"
    )
    print(
        f"    ptrs: objectEvents={hex(obj_ptr)} "
        f"coordEvents={hex(coord_ptr)} bgEvents={hex(bg_ptr)}"
    )

    # Rival template tile: obj[1] gfx=0xf0 flag=0x2D3 tile=(10,3) in BPEF.
    # (7,3) is Birch's tile; prior work was targeting the wrong object.)
    rival_x, rival_y = 10, 3

    objs = _dump_object_events(emu, events_ptr)
    coords = _dump_coord_events(emu, events_ptr, rival_x, rival_y)
    bgs = _dump_bg_events(emu, events_ptr, rival_x, rival_y)

    # ------------------------------------------------------------------
    # Step 4: Dump ROM script bytes for rival objects + nearby events
    # ------------------------------------------------------------------
    print("\n[4] Script bytecode dump ...")

    # Collect rival objects (any gfx in _RIVAL_GFX OR trainerType != 0)
    rival_objs = [o for o in objs if o["gfx"] in _RIVAL_GFX or o["trainer_type"] != 0]
    if not rival_objs:
        # Fall back: gfx 0x40 was seen before; also dump obj[10] if it exists
        rival_objs = [o for o in objs if o["gfx"] == 0x40]
    print(f"  Rival-candidate objects: {[o['idx'] for o in rival_objs]}")
    for o in rival_objs:
        _dump_script_bytes(emu, f"obj[{o['idx']}] gfx=0x{o['gfx']:02x}", o["script_ptr"])

    # Coord/BgEvents near rival
    for c in coords:
        if c["nearby"]:
            _dump_script_bytes(emu, f"coord[{c['idx']}]({c['x']},{c['y']})", c["script_ptr"])
    for b in bgs:
        if b["nearby"]:
            _dump_script_bytes(emu, f"bg[{b['idx']}]({b['x']},{b['y']})", b["script_ptr"])

    # ------------------------------------------------------------------
    # Step 5: ARM ATTEMPT 1 — clear correct flag, rival should now be spawned.
    # Walk adjacent and face+A.
    # ------------------------------------------------------------------
    print("\n[5] ARM ATTEMPT 1: flag 0x2D3 cleared; navigating to rival ...")

    ps = reader.snapshot()
    print(f"    current pos={ps.pos if ps else 'unknown'}")

    walked = _walk_adjacent(emu, reader, rival_x, rival_y)
    print(f"    walk_adjacent -> {walked}")

    started = _face_and_talk(emu, reader, battle, buttons.KEY_UP)
    is_trainer = battle.is_trainer_battle() if started else False
    print(f"    starting={started} trainer={is_trainer}")

    if started and is_trainer:
        Path(OUTPUT_STATE).write_bytes(emu.save_state())
        print(f"\n>>> SUCCESS: saved {OUTPUT_STATE}")
        print(">>> ARMED (artifact minted, gate = FLAG_HIDE_ROUTE_103_RIVAL=0x2D3 cleared)")
        return 0

    # ------------------------------------------------------------------
    # Step 6: ARM ATTEMPT 2 — also check if gfx=0x40 template uses a DIFFERENT
    # flag. Scan for any rival-gfx objects still hidden and clear them too.
    # ------------------------------------------------------------------
    print("\n[6] ARM ATTEMPT 2: reload state fresh, clear ALL rival-gfx flags ...")
    emu.load_state(state_bytes)
    emu.step(0, 4)
    reader = WorldReader(emu.read_bytes)
    battle = BattleReader(emu.read_bytes)

    # Clear 0x2D3 first
    _clear_flag(emu, RIVAL_FLAG)

    # Also reload map to read templates, then clear any still-hidden rival flags
    _reload_route_103(emu, reader)
    emu.step(0, RELEASE_FRAMES * 4)

    events_ptr2 = _u32(emu, _G_MAP_HEADER + _EVENTS_OFF)
    if _ROM_S <= events_ptr2 < _ROM_E:
        obj_ptr2 = _u32(emu, events_ptr2 + _ME_OBJ_PTR)
        obj_count2 = _u8(emu, events_ptr2 + _ME_OBJ_COUNT)
        if _ROM_S <= obj_ptr2 < _ROM_E:
            for i in range(obj_count2):
                base = obj_ptr2 + i * _OBJ_STRIDE
                gfx = _u8(emu, base + _OT_GFX)
                fid = _u16(emu, base + _OT_FLAG)
                if gfx in _RIVAL_GFX and fid != 0:
                    fval = _flag_set(emu, fid)
                    if fval:  # still hidden
                        cleared = _clear_flag(emu, fid)
                        print(f"    also cleared obj[{i}] gfx=0x{gfx:02x} "
                              f"flagId=0x{fid:04x} -> {cleared}")

    # Reload once more so new flag states take effect
    ok2 = _reload_route_103(emu, reader)
    if not ok2:
        print("  reload failed on attempt 2")
    emu.step(0, RELEASE_FRAMES * 4)

    ps2 = reader.snapshot()
    print(f"    pos after reload={ps2.pos if ps2 else 'unknown'}")

    walked2 = _walk_adjacent(emu, reader, rival_x, rival_y)
    print(f"    walk_adjacent -> {walked2}")

    started2 = _face_and_talk(emu, reader, battle, buttons.KEY_UP)
    is_trainer2 = battle.is_trainer_battle() if started2 else False
    print(f"    starting={started2} trainer={is_trainer2}")

    if started2 and is_trainer2:
        Path(OUTPUT_STATE).write_bytes(emu.save_state())
        print(f"\n>>> SUCCESS: saved {OUTPUT_STATE}")
        print(">>> ARMED (artifact minted, gate = all rival-gfx flags cleared)")
        return 0

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print("\n=== FINAL REPORT ===")
    print(f"MapEvents @ {hex(events_ptr)}")
    print(f"  counts: obj={obj_count} warp={warp_count} coord={coord_count} bg={bg_count}")
    print(f"  ptrs: objectEvents={hex(obj_ptr)} coordEvents={hex(coord_ptr)} bgEvents={hex(bg_ptr)}")
    print()
    print("Rival candidate objects:")
    for o in rival_objs:
        print(f"  obj[{o['idx']}] gfx=0x{o['gfx']:02x} tile=({o['x']},{o['y']}) "
              f"trainerType={o['trainer_type']} flagId=0x{o['flag_id']:04x} "
              f"script={hex(o['script_ptr'])}")
    print()
    print(f"ARM attempt 1: starting={started} trainer={is_trainer}")
    print(f"ARM attempt 2: starting={started2} trainer={is_trainer2}")
    print()
    print(">>> GATE-IDENTIFIED-BUT-NOT-ARMED or INCONCLUSIVE — see bytecode above")
    print("    The route_103 rival script (Route103_EventScript_Rival) has NO internal")
    print("    gate beyond the object's hide-flag. If face+A still produces nothing,")
    print("    the object either did not respawn (flag 0x2D3 / template flag not cleared)")
    print("    or the facing direction / stand cell needs adjustment.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
