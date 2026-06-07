import numpy as np

from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv


def test_seeded_reset_is_deterministic():
    env = CartPoleNEnv(CartPoleNConfig(n_poles=2))
    obs_a, _ = env.reset(seed=7)
    obs_b, _ = env.reset(seed=7)
    assert np.allclose(obs_a, obs_b)


def test_step_shapes_for_three_poles():
    env = CartPoleNEnv(CartPoleNConfig(n_poles=3))
    obs, _ = env.reset(seed=1)
    assert obs.shape == (11,)
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (11,)
    assert reward in (0.0, 1.0)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "raw_state" in info

