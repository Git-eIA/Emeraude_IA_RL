"""Throwaway diagnostic: check whether the rival object (gfx=0x40) is live in
gObjectEvents when the player is standing adjacent to (7,3) on route_103.

Resolves the (a)/(b) fork:
  (a) rival NOT spawned even when adjacent  -> extra story flags gate it
  (b) rival IS live at gfx=0x40 near (7,3) -> sprite spawns, talk is not the trigger
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.capture_trainer_battle as cap
from emulator.gba import GbaEmulator
from agent.train_fighter import make_move_type_fn
from env.game_state import BattleReader, SAVE_BLOCK1_PTR
from env.grid_navigator import snapshot_settled
from env.world_reader import WorldReader

# ── ROM map-header chain constants (BPEF == BPEE layout) ───────────────────────
_G_MAP_HEADER = 0x02037318
_EVENTS_OFF = 0x04            # MapHeader.events (ptr)
_OBJ_COUNT_OFF = 0x00         # MapEvents.objectEventCount (u8)
_OBJ_PTR_OFF = 0x04           # MapEvents.objectEvents (ptr)
_TEMPLATE_SIZE = 0x18         # sizeof(ObjectEventTemplate)
_T_GFX = 0x01                 # graphicsId (u8)
_T_X = 0x04                   # s16 x (map tile)
_T_Y = 0x06                   # s16 y (map tile)
_T_TRAINER_TYPE = 0x0E        # u16 trainerType
_T_FLAG = 0x14                # u16 flagId
_FLAGS_OFF = 0x1270           # SaveBlock1.flags[] offset
_ROM_START = 0x08000000
_ROM_END = 0x0A000000
_EWRAM_START = 0x02000000
_EWRAM_END = 0x02040000


def _u8(emu, addr: int) -> int:
    return emu.read_bytes(addr, 1)[0]


def _u16(emu, addr: int) -> int:
    return int.from_bytes(emu.read_bytes(addr, 2), "little")


def _s16(emu, addr: int) -> int:
    return int.from_bytes(emu.read_bytes(addr, 2), "little", signed=True)


def _u32(emu, addr: int) -> int:
    return int.from_bytes(emu.read_bytes(addr, 4), "little")


def _read_flag_bit(emu, flag_id: int) -> int | None:
    """Return the raw bit (0=clear/shown, 1=set/hidden) for flag_id, or None on bad ptr."""
    ptr = _u32(emu, SAVE_BLOCK1_PTR)
    if not _EWRAM_START <= ptr < _EWRAM_END:
        return None
    byte = _u8(emu, ptr + _FLAGS_OFF + flag_id // 8)
    return (byte >> (flag_id % 8)) & 1


def _scan_objects_verbose(emu) -> None:
    """Print all live gObjectEvents slots — same output as cap._scan_objects."""
    blob = emu.read_bytes(0x02037350, 0x24 * 16)
    found_any = False
    for i in range(16):
        o = i * 0x24
        if not blob[o] & 1:
            continue
        found_any = True
        gx = int.from_bytes(blob[o + 0x10:o + 0x12], "little", signed=True)
        gy = int.from_bytes(blob[o + 0x12:o + 0x14], "little", signed=True)
        gfx = blob[o + 5]
        tag = " <-- RIVAL GFX" if gfx == 0x40 else ""
        print(f"    live obj slot={i} gfx=0x{gfx:02x} tile=({gx - 7},{gy - 7}){tag}")
    if not found_any:
        print("    (no live slots found)")


def _walk_map_templates(emu) -> list[dict]:
    """Follow gMapHeader -> MapEvents -> ObjectEventTemplate[] and return all entries."""
    events_ptr = _u32(emu, _G_MAP_HEADER + _EVENTS_OFF)
    print(f"  gMapHeader.events = 0x{events_ptr:08x}")
    if not _ROM_START <= events_ptr < _ROM_END:
        print("  events ptr not in ROM range -- map header not settled")
        return []
    count = _u8(emu, events_ptr + _OBJ_COUNT_OFF)
    templates_ptr = _u32(emu, events_ptr + _OBJ_PTR_OFF)
    print(f"  objectEventCount={count} objectEvents=0x{templates_ptr:08x}")
    if not _ROM_START <= templates_ptr < _ROM_END:
        print("  objectEvents ptr not in ROM range")
        return []
    out = []
    for i in range(count):
        base = templates_ptr + i * _TEMPLATE_SIZE
        out.append({
            "index": i,
            "gfx": _u8(emu, base + _T_GFX),
            "x": _s16(emu, base + _T_X),
            "y": _s16(emu, base + _T_Y),
            "trainerType": _u16(emu, base + _T_TRAINER_TYPE),
            "flagId": _u16(emu, base + _T_FLAG),
        })
    return out


def main() -> int:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    emu = GbaEmulator(rom)
    with open(cap.INPUT_STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)
    reader = WorldReader(emu.read_bytes)

    here = reader.snapshot()
    print(f"start map={here.map_id} pos={here.pos}")

    # ── Step 1: clear flag and confirm ──────────────────────────────────────────
    cap._clear_flag(emu, cap.RIVAL_FLAG)
    bit_after_clear = _read_flag_bit(emu, cap.RIVAL_FLAG)
    print(f"[step1] flag 0x{cap.RIVAL_FLAG:04x} bit after clear = {bit_after_clear}  "
          f"(0=cleared/shown, 1=still hidden)")

    # ── Step 2: reload route_103 ─────────────────────────────────────────────────
    print("[step2] reloading route_103 ...")
    back = cap._reload_route_103(emu, reader)
    if back is None:
        print("[step2] ABORT: _reload_route_103 returned None")
        return 1
    print(f"[step2] back on route_103 pos={back.pos}")

    # ── Step 3: build Fighter and navigate to stand cell ────────────────────────
    battle = BattleReader(emu.read_bytes)
    from stable_baselines3 import PPO  # noqa: PLC0415 -- heavy import, deferred
    model = PPO.load(cap.FIGHTER_CKPT, device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    mtf = make_move_type_fn(emu)
    here = snapshot_settled(reader)
    stand_cell, facing = cap._pick_stand_cell(reader, here)
    if stand_cell is None:
        print(f"[step3] ABORT: no path to a cell adjacent to {cap.RIVAL_TILE}")
        return 1
    print(f"[step3] navigating {here.pos} -> stand_cell={stand_cell} facing={facing}")
    lost = cap._navigate(emu, reader, battle, mtf, predict, stand_cell)
    arrival = snapshot_settled(reader)
    arrival_pos = arrival.pos if arrival else "unknown"
    print(f"[step3] nav outcome={lost!r}  arrival_pos={arrival_pos}")

    # ── Step 4: THE KEY STEP -- scan gObjectEvents from adjacent position ────────
    print("[step4] _scan_objects AT arrival position (camera centred near (7,3)):")
    _scan_objects_verbose(emu)

    # ── Step 5: re-read live flag bit + ROM template chain ─────────────────────
    bit_live = _read_flag_bit(emu, cap.RIVAL_FLAG)
    print(f"[step5] live flag 0x{cap.RIVAL_FLAG:04x} bit = {bit_live}  "
          f"(0=clear/shown, 1=still hidden)")

    print("[step5] ROM map-header template walk:")
    templates = _walk_map_templates(emu)
    for t in templates:
        tag = ""
        if t["gfx"] == 0x40:
            tag += " <-- gfx=0x40 (RIVAL CANDIDATE)"
        if t["trainerType"]:
            tag += " <-- TRAINER"
        print(
            f"  obj[{t['index']}] gfx=0x{t['gfx']:02x} tile=({t['x']},{t['y']}) "
            f"trainerType={t['trainerType']} flagId=0x{t['flagId']:04x}{tag}"
        )
        if t["index"] == 10:
            print(f"  --> obj[10] flagId confirmed = 0x{t['flagId']:04x}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
