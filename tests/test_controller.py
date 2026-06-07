from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig


def test_recon_controller_returns_discrete_action():
    env = CartPoleNEnv(CartPoleNConfig(n_poles=1))
    obs, info = env.reset(seed=0)
    controller = ReConCartPoleController(RunnerConfig(n_poles=1, mode="static_recon"))
    action, diagnostics = controller.act(obs, info["raw_state"])
    assert action in (0, 1)
    assert diagnostics["selected_regime"]

