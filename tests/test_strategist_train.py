"""eval_policy harness: deterministic baselines behave as expected."""
from __future__ import annotations

from agent.train_strategist import always_grind, eval_policy


def test_always_grind_never_clears_and_wastes_the_budget() -> None:
    # Grinding only never advances -> 0 challenges cleared, truncates at the
    # 50-step budget paying STEP_GRIND (-1) each step -> mean reward -50.
    mean_reward, clear_rate = eval_policy(always_grind, episodes=20, seed=0)
    assert clear_rate == 0.0
    assert abs(mean_reward - (-50.0)) < 1e-6
