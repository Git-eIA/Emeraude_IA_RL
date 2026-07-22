"""GBA key bitmask constants, matching the KEYINPUT register bit order.

Note: mgba.gba.GBA.KEY_* constants are *indices* (0–9), not bitmasks.
Our public contract uses bitmasks (1 << index) so callers can combine keys
with bitwise OR: e.g. KEY_A | KEY_B. Conversion: mask = 1 << GBA.KEY_<name>.
"""
from __future__ import annotations

KEY_A = 1 << 0
KEY_B = 1 << 1
KEY_SELECT = 1 << 2
KEY_START = 1 << 3
KEY_RIGHT = 1 << 4
KEY_LEFT = 1 << 5
KEY_UP = 1 << 6
KEY_DOWN = 1 << 7
KEY_R = 1 << 8
KEY_L = 1 << 9
