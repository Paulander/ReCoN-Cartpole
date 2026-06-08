from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from recon_lite import NodeState, ReConEngine
from recon_lite.plasticity import (
    BanditConfig,
    ModulationConfig,
    PlasticityConfig,
    apply_fast_update,
    assign_reward,
    choose_child,
    compute_modulators,
    init_bandit_state,
    init_plasticity_state,
    reset_bandit_episode,
    reset_episode,
    snapshot_bandit,
    snapshot_plasticity,
    update_eligibility,
)

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.control.arbitration import arbitrate_force
from recon_cartpole.control.controllers import ControllerMode, force_to_discrete, heuristic_force, random_action
from recon_cartpole.control.goal_vector import compute_cartpole_goal_vector
from recon_cartpole.control.scripts import REGIMES, ForceProposal, ProposalGains, propose_force_for_regime
from recon_cartpole.control.sensors import StateFeatures, features_from_state

from .fired_edges import fired_edges_from_requests
from .graph_factory import GraphConfig, build_cartpole_graph, trainable_edge_whitelist


@dataclass
class RunnerConfig:
    n_poles: int = 1
    mode: ControllerMode = "static_recon"
    action_mode: str = "discrete"
    force_mag: float = 10.0
    max_engine_ticks: int = 32
    stage: str = "default"
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    modulation: ModulationConfig = field(default_factory=ModulationConfig)
    reset_bandit_each_episode: bool = True
    learn: bool = True
    proposal_gains: ProposalGains = field(default_factory=ProposalGains)


class ReConCartPoleController:
    def __init__(self, config: RunnerConfig | None = None):
        self.config = config or RunnerConfig()
        self.graph = build_cartpole_graph(self.config.n_poles, GraphConfig(self.config.action_mode, self.config.force_mag))
        self.engine = ReConEngine(self.graph)
        self.bandit_state = init_bandit_state({"select_control_regime": list(REGIMES)})
        self.plasticity_state = init_plasticity_state(self.graph, trainable_edge_whitelist(self.graph))
        self.last_selected_regime: str | None = None
        self.last_goal_vector: dict[str, Any] = {}
        self.last_modulators = compute_modulators({}, self.config.modulation)
        self.last_fired_edges: list[dict[str, str]] = []
        self.last_proposal = ForceProposal("none", 0.0, 0.0, 0.0, "not run")

    def start_episode(self) -> None:
        self.engine.reset_states()
        reset_episode(self.plasticity_state, self.graph)
        if self.config.reset_bandit_each_episode:
            reset_bandit_episode(self.bandit_state)
        self.last_selected_regime = None
        self.last_fired_edges = []

    def observe_reward(self, reward: float) -> None:
        if not self.config.learn:
            return
        if self.last_selected_regime and self.config.mode in ("recon_bandit", "recon_fast_bandit", "recon_slow"):
            assign_reward("select_control_regime", self.last_selected_regime, reward, self.bandit_state)
        if self.config.mode in ("recon_fast", "recon_fast_bandit", "recon_slow"):
            update_eligibility(self.plasticity_state, self.last_fired_edges, self.config.plasticity.lambda_decay)
            apply_fast_update(
                self.plasticity_state,
                self.graph,
                reward,
                self.last_modulators.eta_tick_eff,
                self.config.plasticity,
            )

    def act(self, observation: Any, raw_state: Any | None = None) -> tuple[Any, dict[str, Any]]:
        if self.config.mode == "baseline_random":
            action = random_action()
            return action, {"force": self.config.force_mag if action else -self.config.force_mag}

        features = features_from_state(observation, raw_state, self.config.n_poles)
        if self.config.mode == "baseline_heuristic":
            force = heuristic_force(features, self.config.force_mag)
            return force_to_discrete(force), {"force": force, "features": features}

        self.engine.reset_states()
        self.graph.nodes["root_balance"].state = NodeState.REQUESTED
        context: dict[str, Any] = {
            "observation": observation,
            "raw_state": raw_state,
            "features": features,
            "n_poles": self.config.n_poles,
            "force_mag": self.config.force_mag,
            "action_mode": self.config.action_mode,
            "callbacks": self._callbacks(),
            "proposals": [],
        }
        self.last_fired_edges = []
        for _ in range(self.config.max_engine_ticks):
            now_requested = self.engine.step(context)
            self.last_fired_edges.extend(fired_edges_from_requests(self.graph, now_requested))
            if "action" in context:
                break

        action = context.get("action", 0)
        self.last_goal_vector = context.get("goal_vector", {})
        self.last_modulators = compute_modulators(self.last_goal_vector, self.config.modulation)
        self.last_selected_regime = context.get("selected_regime", "recover_worst_pole")
        self.last_proposal = context.get("selected_proposal", ForceProposal("none", 0.0, 0.0, 0.0, "fallback"))
        diagnostics = {
            "force": float(context.get("force", 0.0)),
            "goal_vector": self.last_goal_vector,
            "modulators": self.last_modulators.to_dict(),
            "selected_regime": self.last_selected_regime,
            "proposal": asdict(self.last_proposal),
            "fired_edges": list(self.last_fired_edges),
            "plasticity": snapshot_plasticity(self.plasticity_state),
            "bandit": snapshot_bandit(self.bandit_state),
            "graph_nodes": {nid: node.state.name for nid, node in self.graph.nodes.items()},
        }
        return action, diagnostics

    def _callbacks(self):
        return {
            "observe_state": self._observe_state,
            "estimate_risk": self._estimate_risk,
            "proposal": self._proposal,
            "arbitrate_force": self._arbitrate_force,
            "apply_force": self._apply_force,
            "pole_sensor": self._pole_sensor,
        }

    def _observe_state(self, _node, env):
        env["features"] = features_from_state(env["observation"], env.get("raw_state"), self.config.n_poles)
        return True, True

    def _estimate_risk(self, _node, env):
        env["goal_vector"] = compute_cartpole_goal_vector(
            env["features"],
            n_poles=self.config.n_poles,
            stage=self.config.stage,
        )
        env["modulators"] = compute_modulators(env["goal_vector"], self.config.modulation)
        env["selected_regime"] = self._select_regime(env["features"], env["goal_vector"], env["modulators"].c_explore_eff)
        return True, True

    def _proposal(self, node, env):
        regime = node.meta["regime"]
        selected = env.get("selected_regime")
        proposal = propose_force_for_regime(
            regime,
            env["features"],
            self.config.force_mag,
            self.config.proposal_gains,
        )
        if regime != selected:
            proposal.confidence *= 0.08
            proposal.urgency *= 0.25
        env.setdefault("proposals", []).append(proposal)
        return True, True

    def _arbitrate_force(self, _node, env):
        proposal = arbitrate_force(env.get("proposals", []), self.config.force_mag)
        env["selected_proposal"] = proposal
        env["force"] = proposal.force
        return True, True

    def _apply_force(self, _node, env):
        env["action"] = action_from_force(float(env.get("force", 0.0)), self.config.action_mode)
        return True, True

    def _pole_sensor(self, node, env):
        idx = int(node.meta["pole_index"])
        env.setdefault("pole_sensor_values", {})[idx] = env["features"].poles[idx]
        return True, True

    def _select_regime(self, features: StateFeatures, goal_vector: dict[str, Any], c_explore_eff: float) -> str:
        if self.config.mode in ("recon_bandit", "recon_fast_bandit", "recon_slow"):
            child = choose_child("select_control_regime", self.bandit_state, c_explore_eff, self.config.bandit)
            if child:
                return child
        if abs(features.x) > 1.5:
            return "avoid_rail"
        if goal_vector.get("max_velocity_pressure", 0.0) > 0.65:
            return "damp_energy"
        if self.config.n_poles > 1 and features.worst_pole_index > 0:
            return "stabilize_chain"
        return "recover_worst_pole"

