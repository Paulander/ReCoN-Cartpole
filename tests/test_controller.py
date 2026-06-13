import json
import numpy as np
import pytest

from recon_lite import LinkType
from recon_lite.plasticity import ConsolidationConfig, assign_reward

from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import (
    Pole1FixConfig,
    ReConCartPoleController,
    RescueConfig,
    RunnerConfig,
    SubchainBiasConfig,
)
from recon_cartpole.recon.mingru_terminal import MinGRUPrediction, MinGRUTerminalConfig
from recon_cartpole.recon.subchain_terminal import (
    SharedSubchainTerminal,
    SubchainTerminalConfig,
    save_subchain_terminal_checkpoint,
)
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
    def __init__(self, force=10.0, confidence=1.0, logits=None):
        self.force = force
        self.confidence = confidence
        self.logits = list(logits) if logits is not None else [-1.0, 0.0, 1.0]
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
            logits=list(self.logits),
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


def test_recon_mingru_terminal_passthrough_can_override_final_action():
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
                blend=0.0,
                confidence_floor=0.05,
                passthrough_enabled=True,
                passthrough_confidence_floor=0.8,
            ),
        )
    )
    controller.mingru_terminal = FakeMinGRUTerminal(force=-10.0, confidence=0.95)

    action, diagnostics = controller.act(raw, raw)

    assert action == 0
    assert diagnostics["force"] == -controller.config.force_mag
    assert diagnostics["proposal"]["source_node"] == "mingru_terminal"
    assert "mingru_terminal_passthrough" in diagnostics["proposal"]["reason"]
    assert diagnostics["mingru_passthrough"]["passthrough_applied"] is True
    assert diagnostics["mingru_passthrough"]["passthrough_base_proposal"]["source_node"] == "stabilize_chain"


def test_recon_mingru_terminal_passthrough_respects_confidence_floor():
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
                passthrough_enabled=True,
                passthrough_confidence_floor=0.9,
            ),
        )
    )
    controller.mingru_terminal = FakeMinGRUTerminal(force=-10.0, confidence=0.1)

    _action, diagnostics = controller.act(raw, raw)

    assert diagnostics["proposal"]["source_node"] == "stabilize_chain"
    assert "mingru_terminal_passthrough" not in diagnostics["proposal"]["reason"]
    assert diagnostics["mingru_passthrough"]["passthrough_applied"] is False
    assert diagnostics["mingru_passthrough"]["passthrough_confidence_floor"] == 0.9


def test_recon_mingru_terminal_passthrough_respects_logit_margin_floor():
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
                passthrough_enabled=True,
                passthrough_confidence_floor=0.9,
                passthrough_logit_margin_floor=0.2,
            ),
        )
    )
    controller.mingru_terminal = FakeMinGRUTerminal(
        force=-10.0,
        confidence=0.99,
        logits=[0.0, 1.0, 1.05],
    )

    _action, diagnostics = controller.act(raw, raw)

    assert diagnostics["proposal"]["source_node"] == "stabilize_chain"
    assert "mingru_terminal_passthrough" not in diagnostics["proposal"]["reason"]
    assert diagnostics["mingru_passthrough"]["passthrough_applied"] is False
    assert diagnostics["mingru_passthrough"]["passthrough_logit_margin"] == pytest.approx(0.05)
    assert diagnostics["mingru_passthrough"]["passthrough_logit_margin_floor"] == 0.2


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


def test_feedforward_pole1_fix_boosts_midlink_recovery_proposal():
    raw = [0.0, 0.0, 0.01, 0.18, 0.0, 0.0]
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=2,
            mode="recon_feedforward_terminal_with_pole1_fix",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            pole1_fix=Pole1FixConfig(enabled=True, angle_threshold=0.10),
        )
    )

    _action, diagnostics = controller.act(raw, raw)

    assert diagnostics["proposal"]["source_node"] == "stabilize_chain"
    assert "pole1_fix" in diagnostics["proposal"]["reason"]
    assert controller.learning_mechanisms()["pole1_fix"] is True


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


def test_recurrent_policy_terminal_tracks_state_and_episode_start():
    class FakeRecurrentPolicy:
        def __init__(self):
            self.calls = []

        def predict(self, observation, state=None, episode_start=None, deterministic=True):
            self.calls.append(
                {
                    "state": state,
                    "episode_start": np.asarray(episode_start, dtype=bool).copy(),
                    "deterministic": deterministic,
                }
            )
            next_state = "state_1" if state is None else f"{state}_next"
            return 4, next_state

    raw = np.asarray([0.24, 0.5, 0.01, -0.02, 0.03, -0.04, 0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    policy = FakeRecurrentPolicy()
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=4,
            mode="recon_recurrent_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_recurrent=True,
            policy_terminal_observation_mode="normalized_raw",
        )
    )
    controller.policy_terminal_model = policy

    _action, first = controller.act(raw, raw)
    _action, second = controller.act(raw, raw)
    controller.start_episode()
    _action, third = controller.act(raw, raw)

    assert policy.calls[0]["state"] is None
    assert policy.calls[0]["episode_start"].tolist() == [True]
    assert policy.calls[1]["state"] == "state_1"
    assert policy.calls[1]["episode_start"].tolist() == [False]
    assert policy.calls[2]["state"] is None
    assert policy.calls[2]["episode_start"].tolist() == [True]
    assert first["policy_terminal"]["recurrent"] is True
    assert first["policy_terminal"]["episode_start"] is True
    assert second["policy_terminal"]["episode_start"] is False
    assert third["policy_terminal"]["episode_start"] is True


def test_recon_controller_reports_adjacent_subchain_sensor_values():
    raw = np.asarray([0.0, 0.0, 0.01, -0.02, 0.03, -0.04, 0.5, -0.6, 0.7, -0.8], dtype=np.float32)
    controller = ReConCartPoleController(
        RunnerConfig(n_poles=4, mode="static_recon", discrete_action_bins=5, learn=False)
    )

    _action, diagnostics = controller.act(raw, raw)

    subchains = diagnostics["subchain_sensors"]
    assert set(subchains) == {"0_1", "1_2", "2_3"}
    assert subchains["0_1"]["delta_angle"] == pytest.approx(-0.03)
    assert subchains["1_2"]["delta_velocity"] == pytest.approx(1.3)
    assert subchains["2_3"]["mean_angle"] == pytest.approx(-0.005)



def test_subchain_bias_is_default_off_for_static_recon():
    raw = np.asarray([0.0, 0.0, 0.01, 0.12, -0.01, 0.0, 0.0, 0.0], dtype=np.float32)
    controller = ReConCartPoleController(
        RunnerConfig(n_poles=3, mode="static_recon", discrete_action_bins=5, selection_mode="hard_select", learn=False)
    )

    _action, diagnostics = controller.act(raw, raw)

    assert diagnostics["selected_regime"] == "stabilize_chain"
    assert diagnostics["subchain_bias"] == {}
    assert "subchain_bias" not in diagnostics["proposal"]["reason"]
    assert controller.learning_mechanisms()["subchain_terminal"] is False


def test_subchain_bias_mode_changes_stabilize_chain_force_and_reports_votes():
    raw = np.asarray([0.0, 0.0, 0.01, 0.12, -0.01, 0.0, 0.0, 0.0], dtype=np.float32)
    plain = ReConCartPoleController(
        RunnerConfig(n_poles=3, mode="static_recon", discrete_action_bins=5, selection_mode="hard_select", learn=False)
    )
    biased = ReConCartPoleController(
        RunnerConfig(
            n_poles=3,
            mode="recon_subchain_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            subchain_bias=SubchainBiasConfig(
                blend=1.0,
                mean_angle_gain=0.0,
                mean_velocity_gain=0.0,
                delta_angle_gain=100.0,
                delta_velocity_gain=0.0,
                confidence_boost=0.0,
                urgency_boost=0.0,
            ),
        )
    )

    _plain_action, plain_diag = plain.act(raw, raw)
    biased_action, biased_diag = biased.act(raw, raw)

    assert biased.learning_mechanisms()["subchain_terminal"] is True
    assert biased_diag["selected_regime"] == "stabilize_chain"
    assert "subchain_bias" in biased_diag["proposal"]["reason"]
    assert biased_diag["force"] != pytest.approx(plain_diag["force"])
    assert biased_diag["force"] == pytest.approx(biased_diag["subchain_bias"]["subchain_force"])
    assert biased_diag["subchain_bias"]["applied"] is True
    assert [vote["pair"] for vote in biased_diag["subchain_bias"]["votes"]] == [0, 1]
    assert biased_action in range(5)



def test_learned_subchain_terminal_mode_changes_force_and_reports_votes(tmp_path):
    torch = pytest.importorskip("torch")
    raw = np.asarray([0.0, 0.0, 0.01, 0.12, -0.01, 0.0, 0.0, 0.0], dtype=np.float32)
    checkpoint = tmp_path / "subchain_pair.pt"
    cfg = SubchainTerminalConfig(hidden_size=8, blend=1.0, min_confidence=0.0, min_pair_pressure=0.0)
    terminal = SharedSubchainTerminal(n_poles=3, force_mag=10.0, config=cfg)
    model = terminal.build_model(hidden_size=8)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model[-1].bias[0] = 1.2
        model[-1].bias[1] = 5.0
    save_subchain_terminal_checkpoint(checkpoint, model, cfg)

    plain = ReConCartPoleController(
        RunnerConfig(n_poles=3, mode="static_recon", discrete_action_bins=5, selection_mode="hard_select", learn=False)
    )
    learned = ReConCartPoleController(
        RunnerConfig(
            n_poles=3,
            mode="recon_learned_subchain_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            learned_subchain_terminal=SubchainTerminalConfig(
                checkpoint_path=str(checkpoint),
                blend=1.0,
                min_confidence=0.0,
                min_pair_pressure=0.0,
                confidence_boost=0.0,
                urgency_boost=0.0,
            ),
        )
    )

    _plain_action, plain_diag = plain.act(raw, raw)
    learned_action, learned_diag = learned.act(raw, raw)

    assert learned.learning_mechanisms()["learned_subchain_terminal"] is True
    assert learned_diag["learned_subchain_terminal"]["applied"] is True
    assert "learned_subchain_terminal" in learned_diag["proposal"]["reason"]
    assert learned_diag["force"] != pytest.approx(plain_diag["force"])
    assert learned_diag["force"] == pytest.approx(learned_diag["learned_subchain_terminal"]["subchain_force"])
    assert [vote["pair"] for vote in learned_diag["learned_subchain_terminal"]["votes"]] == [0, 1]
    assert learned_action in range(5)

def test_policy_terminal_normalized_raw_prev_force_observation_mode():
    class FakePolicy:
        def __init__(self):
            self.observations = []

        def predict(self, observation, deterministic=True):
            self.observations.append(np.asarray(observation, dtype=np.float32).copy())
            return 4, None

    raw = np.asarray([0.24, 0.5, 0.01, -0.02, 0.03, -0.04, 0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    policy = FakePolicy()
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=4,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_observation_mode="normalized_raw_prev_force",
        )
    )
    controller.policy_terminal_model = policy

    controller.act(raw, raw)
    controller.act(raw, raw)

    assert policy.observations[0].shape == (11,)
    assert policy.observations[0][-1] == 0.0
    assert policy.observations[1][-1] == 1.0


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


def test_rescue_patches_are_default_off():
    class FakePolicy:
        def predict(self, observation, deterministic=True):
            return 0, None

    raw = [0.0, 0.0, 0.15, 0.01, 5.0, 0.0]
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=2,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
        )
    )
    controller.policy_terminal_model = FakePolicy()
    controller.episode_step = 450

    _action, diagnostics = controller.act(raw, raw)

    assert diagnostics["selected_regime"] == "damp_energy"
    assert "policy_terminal" not in diagnostics["proposal"]["reason"]
    assert diagnostics["rescue"] == {}
    assert controller.learning_mechanisms()["rescue_patches"] is False


def test_terminal_passthrough_rescue_can_take_over_selected_high_risk_regime():
    class FakePolicy:
        def predict(self, observation, deterministic=True):
            return 0, None

    raw = [0.0, 0.0, 0.15, 0.01, 5.0, 0.0]
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=2,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            rescue=RescueConfig(
                enabled=True,
                terminal_force_passthrough_high_confidence=True,
                passthrough_start_step=400,
                passthrough_angle_threshold=0.14,
            ),
        )
    )
    controller.policy_terminal_model = FakePolicy()
    controller.episode_step = 450

    action, diagnostics = controller.act(raw, raw)

    assert diagnostics["selected_regime"] == "damp_energy"
    assert "rescue_passthrough" in diagnostics["proposal"]["reason"]
    assert diagnostics["rescue"]["policy_terminal"]["rescue_passthrough"] is True
    assert "terminal_force_passthrough_high_confidence" in diagnostics["rescue"]["events"]
    assert diagnostics["force"] == -controller.config.force_mag
    assert action == 0
    assert controller.learning_mechanisms()["rescue_patches"] is True


def test_recon_policy_terminal_padded_prev_force_observation_mode():
    class FakePolicy:
        def __init__(self):
            self.observations = []

        def predict(self, observation, deterministic=True):
            self.observations.append(np.asarray(observation, dtype=np.float32).copy())
            return 4, None

    raw = np.asarray([0.24, 0.5, 0.01, -0.02, -0.03, 0.1, -0.2, 0.3], dtype=np.float32)
    policy = FakePolicy()
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=3,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_observation_mode="normalized_raw4_prev_force",
        )
    )
    controller.policy_terminal_model = policy

    controller.act(raw, raw)
    controller.act(raw, raw)

    assert policy.observations[0].shape == (11,)
    assert policy.observations[0][5] == 0.0
    assert policy.observations[0][9] == 0.0
    assert policy.observations[0][-1] == 0.0
    assert policy.observations[1][-1] == 1.0


def test_policy_terminal_normalizer_is_applied(tmp_path):
    class FakePolicy:
        def __init__(self):
            self.observation = None

        def predict(self, observation, deterministic=True):
            self.observation = np.asarray(observation, dtype=np.float32).copy()
            return 4, None

    normalizer_path = tmp_path / "normalizer.json"
    normalizer_path.write_text(
        json.dumps({"mean": [0.0, 0.0, 0.0, 0.0], "var": [1.0, 1.0, 1.0, 1.0], "epsilon": 0.0, "clip_obs": 10.0}),
        encoding="utf-8",
    )
    policy = FakePolicy()
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=1,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_observation_mode="normalized_raw",
            policy_terminal_normalizer_path=str(normalizer_path),
        )
    )
    controller.policy_terminal_model = policy

    raw = np.asarray([2.0, 5.0, 0.0, 0.0], dtype=np.float32)
    _action, diagnostics = controller.act(raw, raw)

    assert np.allclose(policy.observation, [2.0 / 2.4, 1.0, 0.0, 0.0])
    assert diagnostics["policy_terminal"]["normalizer_applied"] is True


def test_residual_policy_terminal_proposal_diagnostics_feature_mode_expands_observation():
    class FakeBasePolicy:
        def predict(self, observation, deterministic=True):
            return 2, None

    class FakeResidualPolicy:
        def __init__(self):
            self.observation = None

        def predict(self, observation, deterministic=True):
            self.observation = np.asarray(observation, dtype=np.float32).copy()
            return 2, None

    raw = np.asarray([0.1, -0.2, 0.05, -0.08, 0.02, 0.1], dtype=np.float32)
    residual = FakeResidualPolicy()
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=2,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_observation_mode="normalized_raw",
            residual_policy_terminal_mode="bin_delta",
            residual_policy_terminal_action_bins=5,
            residual_policy_terminal_gate_threshold=0.0,
            residual_policy_terminal_feature_mode="proposal_diagnostics",
        )
    )
    controller.policy_terminal_model = FakeBasePolicy()
    controller.residual_policy_terminal_model = residual

    _force, info = controller._policy_terminal_force(raw, raw)

    residual_info = info["residual_policy_terminal"]
    assert residual_info["feature_mode"] == "proposal_diagnostics"
    assert residual_info["aux_feature_size"] > 3
    assert residual.observation.shape == (residual_info["observation_size"],)
    assert residual_info["observation_size"] > residual_info["aux_feature_size"]


def test_torch_residual_policy_terminal_can_be_loaded(tmp_path):
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(7, 4), nn.ReLU(), nn.Linear(4, 5))
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model[2].bias[4] = 1.0
    path = tmp_path / "residual.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {"input_size": 7, "hidden_size": 4, "classes": 5},
        },
        path,
    )

    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=1,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            residual_policy_terminal_path=str(path),
        )
    )

    action, _state = controller.residual_policy_terminal_model.predict(np.zeros(7, dtype=np.float32))

    assert int(action) == 4


def test_residual_policy_terminal_subchain_feature_mode_expands_observation():
    from recon_cartpole.control.residual_features import residual_aux_feature_size

    class FakeBasePolicy:
        def predict(self, observation, deterministic=True):
            return 2, None

    class FakeResidualPolicy:
        def __init__(self):
            self.observation = None

        def predict(self, observation, deterministic=True):
            self.observation = np.asarray(observation, dtype=np.float32).copy()
            return 2, None

    raw = np.asarray([0.1, -0.2, 0.05, -0.08, 0.02, 0.1], dtype=np.float32)
    residual = FakeResidualPolicy()
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=2,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_observation_mode="normalized_raw",
            residual_policy_terminal_mode="bin_delta",
            residual_policy_terminal_action_bins=5,
            residual_policy_terminal_gate_threshold=0.0,
            residual_policy_terminal_feature_mode="subchain_diagnostics",
        )
    )
    controller.policy_terminal_model = FakeBasePolicy()
    controller.residual_policy_terminal_model = residual

    _force, info = controller._policy_terminal_force(raw, raw)

    residual_info = info["residual_policy_terminal"]
    assert residual_info["feature_mode"] == "subchain_diagnostics"
    assert residual_info["aux_feature_size"] == residual_aux_feature_size("subchain_diagnostics")
    assert residual.observation.shape == (6 + residual_aux_feature_size("subchain_diagnostics"),)


def test_residual_policy_terminal_bin_delta_changes_policy_force():
    class FakeBasePolicy:
        def predict(self, observation, deterministic=True):
            return 2, None

    class FakeResidualPolicy:
        def __init__(self):
            self.observation = None

        def predict(self, observation, deterministic=True):
            self.observation = np.asarray(observation, dtype=np.float32).copy()
            return 4, None

    raw = np.asarray([0.0, 0.0, 0.18, 0.0], dtype=np.float32)
    residual = FakeResidualPolicy()
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=1,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_observation_mode="normalized_raw",
            residual_policy_terminal_mode="bin_delta",
            residual_policy_terminal_action_bins=5,
            residual_policy_terminal_gate_threshold=0.0,
        )
    )
    controller.policy_terminal_model = FakeBasePolicy()
    controller.residual_policy_terminal_model = residual

    force, info = controller._policy_terminal_force(raw, raw)

    assert force == controller.config.force_mag
    assert info["base_force"] == 0.0
    assert info["residual_policy_terminal"]["available"] is True
    assert info["residual_policy_terminal"]["mode"] == "bin_delta"
    assert info["residual_policy_terminal"]["residual_delta"] == controller.config.force_mag
    assert residual.observation.shape == (7,)


def test_residual_policy_terminal_bin_delta_can_hold_option_across_ticks():
    class FakeBasePolicy:
        def predict(self, observation, deterministic=True):
            return 2, None

    class FakeResidualPolicy:
        def __init__(self):
            self.actions = [4, 2, 2, 2]
            self.calls = 0

        def predict(self, observation, deterministic=True):
            action = self.actions[min(self.calls, len(self.actions) - 1)]
            self.calls += 1
            return action, None

    raw = np.asarray([0.0, 0.0, 0.18, 0.0], dtype=np.float32)
    residual = FakeResidualPolicy()
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=1,
            mode="recon_policy_terminal",
            discrete_action_bins=5,
            selection_mode="hard_select",
            learn=False,
            policy_terminal_observation_mode="normalized_raw",
            residual_policy_terminal_mode="bin_delta",
            residual_policy_terminal_action_bins=5,
            residual_policy_terminal_gate_threshold=0.0,
            residual_policy_terminal_hold_steps=3,
        )
    )
    controller.policy_terminal_model = FakeBasePolicy()
    controller.residual_policy_terminal_model = residual

    first_force, first_info = controller._policy_terminal_force(raw, raw)
    second_force, second_info = controller._policy_terminal_force(raw, raw)
    third_force, third_info = controller._policy_terminal_force(raw, raw)
    fourth_force, fourth_info = controller._policy_terminal_force(raw, raw)

    assert first_force == controller.config.force_mag
    assert first_info["residual_policy_terminal"]["requested_shift"] == 2
    assert first_info["residual_policy_terminal"]["applied_shift"] == 2
    assert first_info["residual_policy_terminal"]["option_remaining"] == 2
    assert first_info["residual_policy_terminal"]["option_reused"] is False

    assert second_force == controller.config.force_mag
    assert second_info["residual_policy_terminal"]["requested_shift"] == 0
    assert second_info["residual_policy_terminal"]["applied_shift"] == 2
    assert second_info["residual_policy_terminal"]["option_remaining"] == 1
    assert second_info["residual_policy_terminal"]["option_reused"] is True

    assert third_force == controller.config.force_mag
    assert third_info["residual_policy_terminal"]["applied_shift"] == 2
    assert third_info["residual_policy_terminal"]["option_remaining"] == 0
    assert third_info["residual_policy_terminal"]["option_reused"] is True

    assert fourth_force == 0.0
    assert fourth_info["residual_policy_terminal"]["requested_shift"] == 0
    assert fourth_info["residual_policy_terminal"]["applied_shift"] == 0
    assert fourth_info["residual_policy_terminal"]["option_reused"] is False
