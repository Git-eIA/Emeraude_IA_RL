# Module index

| Module | Responsibility | Public API | Depends on |
|--------|---------------|------------|------------|
| emulator/buttons.py | GBA key bitmask constants | KEY_A..KEY_L | — |
| emulator/gba.py | Headless mGBA wrapper | GbaEmulator | libmgba-py, buttons |
| env/game_state.py | Emerald RAM parsing | EmeraldReader, PlayerState | — (reader injected) |
| env/rewards.py | Exploration reward | ExplorationTracker | game_state |
| env/pokemon_env.py | Gymnasium env | PokemonEmeraldEnv | game_state, rewards |
| agent/train.py | PPO training entrypoint | main() | env, sb3, torch |
| tools/run_scripted.py | Scripted-input M1 check | CLI | emulator |
| tools/play_interactive.py | Human play + savestate creation | CLI | emulator, pygame |
