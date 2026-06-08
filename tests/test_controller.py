from recon_lite import LinkType
from recon_lite.plasticity import ConsolidationConfig, assign_reward

from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
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
    controller = ReConCartPoleController(RunnerConfig(n_poles=1, mode="static_recon", selection_mode="soft_select"))
    action_low, diagnostics_low = controller.act(obs, raw)

    controller.set_edge_weight("select_control_regime", "avoid_rail", LinkType.SUB, 0.01)
    controller.set_edge_weight("avoid_rail", "avoid_rail_proposal", LinkType.SUB, 0.01)
    controller.set_edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB, 10.0)
    controller.set_edge_weight("recover_worst_pole", "recover_worst_pole_proposal", LinkType.SUB, 10.0)
    action_high, diagnostics_high = controller.act(obs, raw)

    low_scores = {item["source_node"]: item["score"] for item in diagnostics_low["proposals"]}
    high_scores = {item["source_node"]: item["score"] for item in diagnostics_high["proposals"]}
    assert high_scores["recover_worst_pole"] > low_scores["recover_worst_pole"] * 50.0
    assert diagnostics_high["proposal"]["source_node"] == "recover_worst_pole"
    assert action_low != action_high


def test_hard_select_suppresses_non_selected_proposals():
    raw = [2.0, 0.0, 0.10, 0.0]
    controller = ReConCartPoleController(RunnerConfig(n_poles=1, mode="static_recon", selection_mode="hard_select"))
    _, diagnostics = controller.act(raw, raw)
    assert diagnostics["selected_regime"] == "avoid_rail"
    assert {item["source_node"] for item in diagnostics["proposals"]} == {"avoid_rail"}
    assert any(item["source_node"] == "recover_worst_pole" for item in diagnostics["suppressed_proposals"])


def test_soft_select_includes_all_proposals_with_weight_fields():
    raw = [0.0, 0.0, 0.10, 0.0]
    controller = ReConCartPoleController(RunnerConfig(n_poles=1, mode="static_recon", selection_mode="soft_select"))
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
    for regime in ["avoid_rail", "damp_energy", "recover_worst_pole", "recover_base_pole", "stabilize_chain", "center_cart"]:
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
        consolidation=ConsolidationConfig(enabled=True, min_episodes=2, eta_consolidate=1.0, outcome_weight=1.0),
    )
    controller = ReConCartPoleController(cfg)
    key = "select_control_regime->recover_worst_pole:SUB"
    before = controller.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB)

    controller.plasticity_state[key].delta_sum = 1.0
    first = controller.end_episode([1.0], total_return=1.0, horizon=1)
    assert first["applied"] == {}
    assert controller.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB) == before

    controller.plasticity_state[key].delta_sum = 1.0
    second = controller.end_episode([1.0], total_return=1.0, horizon=1)
    assert key in second["applied"]
    after = controller.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB)
    assert after > before

    controller.set_edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB, after + 0.5)
    controller.start_episode()
    assert controller.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB) == after

    checkpoint = tmp_path / "slow.json"
    controller.save_consolidation_checkpoint(str(checkpoint))
    loaded = ReConCartPoleController(cfg)
    loaded.load_consolidation_checkpoint(str(checkpoint))
    assert loaded.edge_weight("select_control_regime", "recover_worst_pole", LinkType.SUB) == after
