from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.pokemon_env import PokemonEmeraldEnv
from tests.conftest import FakeEmulator


def test_ppo_learns_without_crashing():
    """256 steps of PPO on the fake emulator: wiring works end to end."""
    vec = DummyVecEnv(
        [lambda: PokemonEmeraldEnv(FakeEmulator(), initial_state=b"fake", max_steps=64)]
    )
    model = PPO("CnnPolicy", vec, n_steps=64, batch_size=64, device="cpu", verbose=0)
    model.learn(total_timesteps=256)
