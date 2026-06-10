import numpy as np

from recon_lite import LinkType
from recon_lite.plasticity import ConsolidationConfig, assign_reward

from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.mingru_terminal import MinGRUPrediction, MinGRUTerminalConfig
from recon_cartpole.training.evaluate import rollout


def test_recon_controller_returns_discrete_action():
    env = CartPoleNEnv(CartPoleNConfig(n_poles=1))
    obs, info = env.reset(seed=0)
    controller = ReConCartPoleController(RunnerConfig(n_poles=1, mode="static_recon"))
    action, diagnostics = controller.act(obs, info["raw_state"])
    assert action in (0, 1)
    assert diagnostics["selected_regime"]


def test_trainable_edge_weight_changes_proposal_score_and_action():
    raw = [2.0, 0.0, 0.10, 0.0]
    obs = raw
    controller = ReConCartPoleController(
        RunnerConfig(n_poles=1, mode="static_recon", selection_mode="soft_select")
    )
    action_low, diagnostics_low = controller.act(obs, raw)

    controller.set_edge_weight("select_control_regime", "avoid_rail", LinkType.SUB, 0.01)
    controller.set_edge_weight("avoid_rail", "avoid_rail_proposal", LinkType.SUB, 0.01)
    controller.set_edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB, 10.0)
    controller.set_edge_weight(
        "recover_worst_pole", "recover_worst_pole_proposal", LinkType.SUB, 10.0
    )
    action_high, diagnostics_high = controller.act(obs, raw)

    low_scores = {item["source_node"]: item["score"] for item in diagnostics_low["proposals"]}
    high_scores = {item["source_node"]: item["score"] for item in diagnostics_high["proposals"]}
    assert high_scores["recover_worst_pole"] > low_scores["recover_worst_pole"] * 50.0
    assert diagnostics_high["proposal"]["source_node"] == "recover_worst_pole"
    assert action_low != action_high


def test_hard_select_suppresses_non_selected_proposals():
    raw = [2.0, 0.0, 0.10, 0.0]
    controller = ReConCartPoleController(
        RunnerConfig(n_poles=1, mode="static_recon", selection_mode="hard_select")
    )
    _, diagnostics = controller.act(raw, raw)
    assert diagnostics["selected_regime"] == "avoid_rail"
    assert {item["source_node"] for item in diagnostics["proposals"]} == {"avoid_rail"}
    assert any(
        item["source_node"] == "recover_worst_pole" for item in diagnostics["suppressed_proposals"]
    )


def test_soft_select_includes_all_proposals_with_weight_fields():
    raw = [0.0, 0.0, 0.10, 0.0]
    controller = ReConCartPoleController(
        RunnerConfig(n_poles=1, mode="static_recon", selection_mode="soft_select")
    )
    controller.set_edge_weight("select_control_regime", "center_cart", LinkType.SUB, 0.2)
    _, diagnostics = controller.act(raw, raw)
    sources = {item["source_node"] for item in diagnostics["proposals"]}
    assert "center_cart" in sources
    assert "recover_worst_pole" in sources
    center = next(item for item in diagnostics["proposals"] if item["source_node"] == "center_cart")
    assert center["select_edge_weight"] == 0.2
    assert center["selection_mode"] == "soft_select"


def test_plasticity_deltas_appear_in_trace_after_nonzero_reward():
    env = CartPoleNEnv(CartPoleNConfig(n_poles=1, horizon=4))
    controller = ReConCartPoleController(RunnerConfig(n_poles=1, mode="recon_fast", learn=True))
    result = rollout(env, controller, seed=2, horizon=4, trace=True)
    assert result["trace"]
    assert any(step.get("plasticity") or step.get("fast_deltas") for step in result["trace"])


def test_bandit_rewards_affect_later_regime_choice_when_persistent():
    controller = ReConCartPoleController(
        RunnerConfig(n_poles=1, mode="recon_bandit", reset_bandit_each_episode=False, learn=True)
    )
    for regime in [
        "avoid_rail",
        "damp_energy",
        "recover_worst_pole",
        "recover_base_pole",
        "stabilize_chain",
        "center_cart",
    ]:
        assign_reward("select_control_regime", regime, -1.0, controller.bandit_state)
    assign_reward("select_control_regime", "center_cart", 5.0, controller.bandit_state)
    raw = [0.0, 0.0, 0.01, 0.0]
    _, diagnostics = controller.act(raw, raw)
    assert diagnostics["selected_regime"] == "center_cart"


def test_recon_slow_persists_baseline_changes_after_threshold_and_fast_resets(tmp_path):
    cfg = RunnerConfig(
        n_poles=1,
        mode="recon_slow",
        learn=True,
        consolidation=ConsolidationConfig(
            enabled=True, min_episodes=2, eta_consolidate=1.0, outcome_weight=1.0
        ),
    )
    controller = ReConCartPoleController(cfg)
    key = "select_control_regime->recover_worst_pole:SUB"
    before = controller.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB)

    controller.plasticity_state[key].delta_sum = 1.0
    first = controller.end_episode([1.0], total_return=1.0, horizon=1)
    assert first["applied"]["edges"] == {}
    assert (
        controller.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB)
        == before
    )

    controller.plasticity_state[key].delta_sum = 1.0
    second = controller.end_episode([1.0], total_return=1.0, horizon=1)
    assert key in second["applied"]["edges"]
    after = controller.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB)
    assert after > before

    controller.set_edge_weight(
        "select_control_regime", "recover_worst_pole", LinkType.SUB, after + 0.5
    )
    controller.start_episode()
    assert (
        controller.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB) == after
    )

    checkpoint = tmp_path / "slow.json"
    controller.save_consolidation_checkpoint(str(checkpoint))
    loaded = ReConCartPoleController(cfg)
    loaded.load_consolidation_checkpoint(str(checkpoint))
    assert loaded.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB) == after


def test_recon_learn_only_node_params_affect_force_and_trace():
    raw = [0.0, 0.0, 0.10, 0.0]
    controller = ReConCartPoleController(RunnerConfig(n_poles=1, mode="recon_learn_only"))
    _, before = controller.act(raw, raw)
    base_force = before["force"]
    controller.node_param_state["recover_worst_pole"].current.force_bias = -8.0
    _, after = controller.act(raw, raw)
    assert after["force"] < base_force
    proposal = next(
        item for item in after["proposals"] if item["source_node"] == "recover_worst_pole"
    )
    assert "node_params" in proposal["reason"]
    assert after["node_params"]


def test_recon_learn_only_updates_node_param_deltas_in_trace():
    env = CartPoleNEnv(CartPoleNConfig(n_poles=1, horizon=5))
    controller = ReConCartPoleController(
        RunnerConfig(n_poles=1, mode="recon_learn_only", learn=True)
    )
    result = rollout(env, controller, seed=5, horizon=5, trace=True)
    assert any(step.get("node_param_deltas") for step in result["trace"])


def test_controller_checkpoint_reloads_edge_and_node_params(tmp_path):
    controller = ReConCartPoleController(RunnerConfig(n_poles=1, mode="recon_learn_only"))
    controller.set_edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB, 2.5)
    controller.node_param_state["recover_worst_pole"].base.force_bias = 1.25
    controller.node_param_state["recover_worst_pole"].current.force_bias = 1.25
    checkpoint = tmp_path / "controller.json"
    controller.save_checkpoint(str(checkpoint))

    loaded = ReConCartPoleController(RunnerConfig(n_poles=1, mode="recon_learn_only"))
    loaded.load_consolidation_checkpoint(str(checkpoint))
    assert loaded.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB) == 2.5
    assert loaded.node_param_state["recover_worst_pole"].base.force_bias == 1.25


def test_controller_quantizes_force_to_five_discrete_bins():
    controller = ReConCartPoleController(
        RunnerConfig(n_poles=1, mode="static_recon", discrete_action_bins=5)
    )
    raw = [0.0, 0.0, 0.01, 0.0]
    action, diagnostics = controller.act(raw, raw)
    assert 0 <= action < 5
    assert diagnostics["force"] != 0.0 or action == 2


def test_recon_mlp_terminal_affects_chain_proposal_and_updates():
    raw = [0.0, 0.0, 0.03, -0.02, 0.04, 0.01, 0.1, -0.2, 0.3, -0.1]
    controller = ReConCartPoleController(RunnerConfig(n_poles=4, mode="recon_mlp_terminal"))
    _, diagnostics = controller.act(raw, raw)
    chain = next(
        item for item in diagnostics["proposals"] if item["source_node"] == "stabilize_chain"
    )
    assert "mlp_terminal" in chain["reason"]
    assert diagnostics["mlp_terminal"]["hidden_size"] == controller.config.mlp_terminal.hidden_size

    env = CartPoleNEnv(CartPoleNConfig(n_poles=4, horizon=5, discrete_action_bins=5))
    result = rollout(env, controller, seed=12, horizon=5, trace=True)
    update = result["episode_learning"]["applied"]["mlp_terminal"]
    assert "update_norm" in update
    assert any(step.get("mlp_terminal") for step in result["trace"])


class FakeMinGRUTerminal:
    def __init__(self, force=10.0, confidence=1.0):
        self.force = force
        self.confidence = confidence
        self.calls = 0
        self.reset_calls = 0
        self.loaded_checkpoint = "fake.pt"

    def predict(self, observation, raw_state, context=None):
        self.calls += 1
        return MinGRUPrediction(
            force=self.force,
            confidence=self.confidence,
            value=0.25,
            failure_probability=0.1,
            hidden_norm=1.5,
            sequence_length=4,
            logits=[-1.0, 0.0, 1.0],
        )

    def reset(self):
        self.reset_calls += 1


def _controller_with_fake_mingru(config):
    controller = ReConCartPoleController(config)
    controller.config.mode = "recon_mingru_terminal"
    controller.config.mingru_terminal.enabled = True
    controller.mingru_terminal = FakeMinGRUTerminal()
    return controller


def test_recon_mingru_terminal_can_drive_stabilize_chain_proposal():
    raw = [0.0, 0.0, 0.01, 0.04, 0.12, -0.03, 0.0, 0.0, 0.0, 0.0]
    controller = _controller_with_fake_mingru(
        RunnerConfig(
            n_poles=4,
            mode="static_recon",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            mingru_terminal=MinGRUTerminalConfig(
                enabled=True,
                scope="stabilize_chain",
                blend=1.0,
                confidence_floor=0.05,
            ),
        )
    )

    action, diagnostics = controller.act(raw, raw)

    assert diagnostics["selected_regime"] == "stabilize_chain"
    assert diagnostics["proposal"]["source_node"] == "stabilize_chain"
    assert "mingru_terminal" in diagnostics["proposal"]["reason"]
    assert diagnostics["force"] == controller.config.force_mag
    assert action == 4
    assert diagnostics["mingru_terminal"]["available"] is True
    assert diagnostics["mingru_terminal"]["applied"] is True
    assert diagnostics["mingru_terminal"]["hidden_norm"] == 1.5


def test_recon_mingru_terminal_low_confidence_is_traced_and_downweighted():
    raw = [0.0, 0.0, 0.01, 0.04, 0.12, -0.03, 0.0, 0.0, 0.0, 0.0]
    controller = _controller_with_fake_mingru(
        RunnerConfig(
            n_poles=4,
            mode="static_recon",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            mingru_terminal=MinGRUTerminalConfig(
                enabled=True,
                scope="stabilize_chain",
                blend=1.0,
                confidence_floor=0.5,
            ),
        )
    )
    controller.mingru_terminal = FakeMinGRUTerminal(force=10.0, confidence=0.1)

    _action, diagnostics = controller.act(raw, raw)
    chain = diagnostics["proposal"]

    assert "mingru_terminal" not in chain["reason"]
    assert diagnostics["mingru_terminal"]["applied"] is False
    assert chain["raw_confidence"] < 0.65
    assert diagnostics["mingru_terminal"]["confidence"] == 0.1


def test_recon_mingru_terminal_scope_all_caches_one_prediction_and_resets():
    raw = [0.0, 0.0, 0.02, 0.0]
    controller = _controller_with_fake_mingru(
        RunnerConfig(
            n_poles=1,
            mode="static_recon",
            discrete_action_bins=5,
            selection_mode="soft_select",
            learn=False,
            mingru_terminal=MinGRUTerminalConfig(enabled=True, scope="all"),
        )
    )
    fake = FakeMinGRUTerminal(force=10.0, confidence=1.0)
    controller.mingru_terminal = fake

    _, diagnostics = controller.act(raw, raw)

    assert fake.calls == 1
    applied = {
        item["source_node"] for item in diagnostics["proposals"] if "mingru_terminal" in item["reason"]
    }
    assert applied == {
        "avoid_rail",
        "damp_energy",
        "recover_worst_pole",
        "recover_base_pole",
        "stabilize_chain",
        "center_cart",
    }
    assert set(diagnostics["mingru_terminal"]["applied_regimes"]) == applied

    controller.start_episode()
    assert fake.reset_calls == 1


def test_recon_mingru_plus_learning_enables_recon_learning_mechanisms():
    controller = _controller_with_fake_mingru(
        RunnerConfig(
            n_poles=1,
            mode="static_recon",
            discrete_action_bins=5,
            learn=True,
            reset_bandit_each_episode=False,
            mingru_terminal=MinGRUTerminalConfig(enabled=True),
        )
    )
    controller.config.mode = "recon_mingru_terminal_plus_recon_learning"

    mechanisms = controller.learning_mechanisms()

    assert mechanisms["minGRU_terminal"] is True
    assert mechanisms["edge_plasticity"] is True
    assert mechanisms["bandit_persistence"] is True
    assert mechanisms["slow_consolidation"] is True
    assert mechanisms["node_param_learning"] is False


def test_recon_policy_terminal_can_drive_stabilize_chain_proposal():
    class FakePolicy:
        def predict(self, observation, deterministic=True):
            assert deterministic is True
            return 4, None

    raw = [0.0, 0.0, 0.01, 0.04, 0.12, -0.03, 0.0, 0.0, 0.0, 0.0]
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=4,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
        )
    )
    controller.policy_terminal_model = FakePolicy()
    action, diagnostics = controller.act(raw, raw)
    assert diagnostics["selected_regime"] == "stabilize_chain"
    assert diagnostics["proposal"]["source_node"] == "stabilize_chain"
    assert "policy_terminal" in diagnostics["proposal"]["reason"]
    assert diagnostics["force"] == controller.config.force_mag
    assert action == 4
    assert diagnostics["policy_terminal"]["available"] is True
    assert diagnostics["policy_terminal"]["blend"] == 1.0


def test_recon_policy_terminal_frame_stack_feeds_policy_and_resets():
    class FakePolicy:
        def __init__(self):
            self.observations = []

        def predict(self, observation, deterministic=True):
            assert deterministic is True
            self.observations.append(np.asarray(observation, dtype=np.float32).copy())
            return 4, None

    raw = [0.0, 0.0, 0.01, 0.04, 0.12, -0.03, 0.0, 0.0, 0.0, 0.0]
    obs1 = np.arange(14, dtype=np.float32)
    obs2 = obs1 + 100.0
    policy = FakePolicy()
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=4,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_frame_stack=3,
        )
    )
    controller.policy_terminal_model = policy

    controller.act(obs1, raw)
    _, diagnostics = controller.act(obs2, raw)

    assert policy.observations[0].shape == (42,)
    np.testing.assert_array_equal(
        policy.observations[0].reshape(3, 14), np.stack([obs1, obs1, obs1])
    )
    np.testing.assert_array_equal(
        policy.observations[1].reshape(3, 14), np.stack([obs1, obs1, obs2])
    )
    assert diagnostics["policy_terminal"]["frame_stack"] == 3
    assert diagnostics["policy_terminal"]["observation_size"] == 42

    controller.start_episode()
    controller.act(obs2, raw)
    np.testing.assert_array_equal(
        policy.observations[-1].reshape(3, 14), np.stack([obs2, obs2, obs2])
    )


def test_recon_policy_terminal_normalized_raw_observation_mode():
    class FakePolicy:
        def __init__(self):
            self.observation = None

        def predict(self, observation, deterministic=True):
            self.observation = np.asarray(observation, dtype=np.float32).copy()
            return 4, None

    raw = np.asarray([0.24, 0.5, 0.01, -0.02, 0.03, -0.04, 0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    env_obs = np.arange(14, dtype=np.float32)
    policy = FakePolicy()
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=4,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_observation_mode="normalized_raw",
        )
    )
    controller.policy_terminal_model = policy

    _, diagnostics = controller.act(env_obs, raw)

    assert policy.observation.shape == (10,)
    assert np.isclose(policy.observation[0], 0.1)
    assert np.isclose(policy.observation[1], 0.1)
    assert diagnostics["policy_terminal"]["observation_mode"] == "normalized_raw"
    assert diagnostics["policy_terminal"]["observation_size"] == 10


def test_recon_policy_terminal_scope_all_uses_one_cached_policy_call_for_all_regimes():
    class FakePolicy:
        def __init__(self):
            self.observations = []

        def predict(self, observation, deterministic=True):
            self.observations.append(np.asarray(observation, dtype=np.float32).copy())
            return 4, None

    raw = [0.0, 0.0, 0.02, 0.0]
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=1,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="soft_select",
            learn=False,
            policy_terminal_scope="all",
        )
    )
    policy = FakePolicy()
    controller.policy_terminal_model = policy

    _, diagnostics = controller.act(raw, raw)

    assert len(policy.observations) == 1
    policy_regimes = {
        item["source_node"]
        for item in diagnostics["proposals"]
        if "policy_terminal" in item["reason"]
    }
    assert policy_regimes == {
        "avoid_rail",
        "damp_energy",
        "recover_worst_pole",
        "recover_base_pole",
        "stabilize_chain",
        "center_cart",
    }
    assert diagnostics["policy_terminal"]["scope"] == "all"
    assert set(diagnostics["policy_terminal"]["applied_regimes"]) == policy_regimes


def test_recon_policy_terminal_blend_can_preserve_base_force():
    class FakePolicy:
        def predict(self, observation, deterministic=True):
            return 4, None

    raw = [0.0, 0.0, 0.01, 0.04, 0.12, -0.03, 0.0, 0.0, 0.0, 0.0]
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=4,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_blend=0.0,
        )
    )
    controller.policy_terminal_model = FakePolicy()
    _, diagnostics = controller.act(raw, raw)
    info = diagnostics["policy_terminal"]
    assert info["blend"] == 0.0
    assert diagnostics["force"] == info["base_force"]
    assert info["policy_force"] == controller.config.force_mag
