# Module index

| Module | Responsibility | Public API | Depends on |
|--------|---------------|------------|------------|
| emulator/buttons.py | GBA key bitmask constants | KEY_A..KEY_L | — |
| emulator/gba.py | Headless mGBA wrapper | GbaEmulator | libmgba-py, buttons |
| env/game_state.py | Emerald RAM parsing | EmeraldReader (player_state, read_flag, party_levels), PlayerState | — (reader injected) |
| env/milestones.py | One-shot story milestone rewards | Milestone, MilestoneTracker, starter_milestones | game_state |
| env/rewards.py | Exploration reward | ExplorationTracker, LevelRewardTracker | game_state |
| env/pokemon_env.py | Gymnasium env | PokemonEmeraldEnv | game_state, rewards, milestones |
| agent/train.py | PPO training entrypoint | main() | env, sb3, torch |
| tools/smoke_mgba.py | libmgba-py binding smoke test | CLI | libmgba-py |
| tools/run_scripted.py | Scripted-input M1 check | CLI | emulator |
| tools/play_interactive.py | Human play + savestate creation | CLI | emulator, pygame |
