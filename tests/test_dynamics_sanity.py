from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv


def test_energy_does_not_explode_without_damping_or_force():
    env = CartPoleNEnv(CartPoleNConfig(n_poles=1, damping=0.0, force_mag=0.0))
    env.reset(seed=3)
    start = env.energy()
    for _ in range(40):
        env.step(0)
    assert abs(env.energy() - start) < 0.25

