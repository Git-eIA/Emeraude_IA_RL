# Emu — Pokémon Emerald RL

A PPO agent that learns to play Pokémon Emerald through a headless mGBA core.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    # libmgba-py: see docs/architecture/README.md (built separately)

Provide your own legally dumped Emerald ROM:

    export POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba

## Usage

    python tools/play_interactive.py     # play, create states/initial.state
    python agent/train.py                # train PPO
    tensorboard --logdir runs/

## Tests

    pytest            # ROM-dependent tests skip if POKEMON_EMERALD_ROM unset
