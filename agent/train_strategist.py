"""Train the Strategist: PPO MlpPolicy on the abstract StrategistEnv.

The sim is CPU-cheap (no emulator), so a single env is fine. We also evaluate
two deterministic baselines in the same sim — always-ADVANCE (loses early,
under-leveled) and always-GRIND (wastes the whole budget) — so we can confirm
the learned policy beats both.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.strategist_env import ADVANCE, CHALLENGE_LEVELS, GRIND, StrategistEnv

Policy = Callable[[np.ndarray], int]


def always_advance(_obs: np.ndarray) -> int:
    return ADVANCE


def always_grind(_obs: np.ndarray) -> int:
    return GRIND


def eval_policy(policy: Policy, episodes: int = 100, seed: int = 0) -> tuple[float, float]:
    """Run policy over episodes; return (mean episode reward, clear rate)."""
    env = StrategistEnv()
    total = 0.0
    cleared = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            ep_reward += reward
            done = terminated or truncated
        total += ep_reward
        if env.challenge_idx >= len(CHALLENGE_LEVELS):
            cleared += 1
    return total / episodes, cleared / episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--n-steps", type=int, default=2048)  # steps per PPO rollout
    args = ap.parse_args()

    # Monitor records episode rewards/lengths so SB3 logs rollout/ep_rew_mean
    # and rollout/ep_len_mean (needed to see if the Strategist is learning).
    # Create the checkpoint dir up front: model.save at the end would raise if
    # a short run never hit the first save_freq checkpoint that creates it.
    Path("checkpoints/strategist").mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(StrategistEnv())])
    ckpt = CheckpointCallback(
        save_freq=50_000, save_path="checkpoints/strategist", name_prefix="ppo_strategist"
    )
    model = PPO("MlpPolicy", env, n_steps=args.n_steps, verbose=1, device="cpu")

    adv_reward, adv_clear = eval_policy(always_advance)
    grind_reward, grind_clear = eval_policy(always_grind)
    print(f"baseline always-ADVANCE: mean_reward={adv_reward:.1f} clear_rate={adv_clear:.2f}")
    print(f"baseline always-GRIND:   mean_reward={grind_reward:.1f} clear_rate={grind_clear:.2f}")

    model.learn(total_timesteps=args.timesteps, callback=ckpt)
    model.save("checkpoints/strategist/ppo_strategist_final")

    def trained(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    tr_reward, tr_clear = eval_policy(trained)
    print(f"trained Strategist:      mean_reward={tr_reward:.1f} clear_rate={tr_clear:.2f}")


if __name__ == "__main__":
    main()
