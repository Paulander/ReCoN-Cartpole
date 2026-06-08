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




def test_five_bin_discrete_action_space_maps_to_force_levels():
    env = CartPoleNEnv(CartPoleNConfig(n_poles=1, discrete_action_bins=5))
    assert env.action_space.n == 5
    assert env._force_from_action(0) == -env.config.force_mag
    assert env._force_from_action(2) == 0.0
    assert env._force_from_action(4) == env.config.force_mag


def test_serial_lagrange_dynamics_is_finite_and_distinct():
    parallel = CartPoleNEnv(CartPoleNConfig(n_poles=4, dynamics_mode="parallel"))
    serial = CartPoleNEnv(CartPoleNConfig(n_poles=4, dynamics_mode="serial_lagrange"))
    state = np.asarray([0.0, 0.0, 0.02, -0.03, 0.04, -0.01, 0.0, 0.0, 0.0, 0.0], dtype=float)
    parallel.state = state.copy()
    serial.state = state.copy()
    parallel._integrate(5.0)
    serial._integrate(5.0)
    assert np.all(np.isfinite(serial.state))
    assert not np.allclose(parallel.state, serial.state)
