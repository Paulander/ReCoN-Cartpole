from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from recon_lite import LinkType, NodeState, ReConEngine
from recon_lite.plasticity import (
    BanditConfig,
    ConsolidationConfig,
    ConsolidationEngine,
    EpisodeSummary,
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
    ucb_score,
    update_eligibility,
)

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.control.arbitration import arbitrate_force
from recon_cartpole.control.controllers import ControllerMode, heuristic_force, random_action
from recon_cartpole.control.goal_vector import compute_cartpole_goal_vector
from recon_cartpole.control.scripts import (
    REGIMES,
    ForceProposal,
    ProposalGains,
    propose_force_for_regime,
)
from recon_cartpole.control.sensors import StateFeatures, features_from_state

from .fired_edges import fired_edges_from_requests
from .graph_factory import GraphConfig, build_cartpole_graph, trainable_edge_whitelist
from .mlp_terminal import MlpTerminalConfig, MlpTerminalState
from .node_params import (
    NodeParamConfig,
    RegimeParamState,
    apply_node_params,
    consolidate_regime_params,
    init_regime_param_state,
    snapshot_regime_params,
    update_regime_param_state,
)


@dataclass
class RunnerConfig:
    n_poles: int = 1
    mode: ControllerMode = "static_recon"
    action_mode: str = "discrete"
    force_mag: float = 10.0
    discrete_action_bins: int = 2
    max_engine_ticks: int = 32
    stage: str = "default"
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    modulation: ModulationConfig = field(default_factory=ModulationConfig)
    reset_bandit_each_episode: bool = True
    learn: bool = True
    proposal_gains: ProposalGains = field(default_factory=ProposalGains)
    selection_mode: str = "soft_select"
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    node_params: NodeParamConfig = field(default_factory=NodeParamConfig)
    mlp_terminal: MlpTerminalConfig = field(default_factory=MlpTerminalConfig)
    policy_terminal_path: str = ""
    policy_terminal_blend: float = 1.0
    policy_terminal_frame_stack: int = 1


class ReConCartPoleController:
    def __init__(self, config: RunnerConfig | None = None):
        self.config = config or RunnerConfig()
        self.graph = build_cartpole_graph(
            self.config.n_poles, GraphConfig(self.config.action_mode, self.config.force_mag)
        )
        self.engine = ReConEngine(self.graph)
        self.bandit_state = init_bandit_state({"select_control_regime": list(REGIMES)})
        self.plasticity_state = init_plasticity_state(
            self.graph, trainable_edge_whitelist(self.graph)
        )
        if self._uses_node_params() and not self.config.node_params.enabled:
            self.config.node_params.enabled = True
        self.node_param_state = init_regime_param_state()
        if self._uses_slow_consolidation() and not self.config.consolidation.enabled:
            self.config.consolidation.enabled = True
        if self._uses_mlp_terminal() and not self.config.mlp_terminal.enabled:
            self.config.mlp_terminal.enabled = True
        self.mlp_terminal_state = MlpTerminalState.create(
            self.config.n_poles, self.config.mlp_terminal.hidden_size
        )
        self.mlp_rng = np.random.default_rng(1009 + self.config.n_poles)
        self.last_mlp_terminal: dict[str, Any] = {}
        self.policy_terminal_model: Any | None = None
        self.policy_terminal_obs_history: list[np.ndarray] = []
        self.last_policy_terminal: dict[str, Any] = {}
        if self._uses_policy_terminal() and self.config.policy_terminal_path:
            self.policy_terminal_model = self._load_policy_terminal(
                self.config.policy_terminal_path
            )
        self.consolidation = ConsolidationEngine(self.config.consolidation)
        self.consolidation.init_from_graph(self.graph, list(self.plasticity_state))
        self.last_selected_regime: str | None = None
        self.last_goal_vector: dict[str, Any] = {}
        self.last_modulators = compute_modulators({}, self.config.modulation)
        self.last_fired_edges: list[dict[str, str]] = []
        self.last_fast_deltas: dict[str, float] = {}
        self.last_node_param_deltas: dict[str, float] = {}
        self.last_consolidation_applied: dict[str, Any] = {}
        self.last_proposal = ForceProposal("none", 0.0, 0.0, 0.0, "not run")

    def _uses_fast_plasticity(self) -> bool:
        return self.config.mode in (
            "recon_fast",
            "recon_fast_bandit",
            "recon_slow",
            "gain_search_recon_fast_bandit",
            "recon_learn_only",
            "recon_slow_no_gain_search",
            "recon_mlp_terminal",
        )

    def _uses_bandit(self) -> bool:
        return self.config.mode in (
            "recon_bandit",
            "recon_fast_bandit",
            "recon_slow",
            "gain_search_recon_fast_bandit",
            "recon_learn_only",
            "recon_slow_no_gain_search",
            "recon_mlp_terminal",
        )

    def _uses_slow_consolidation(self) -> bool:
        return self.config.mode in (
            "recon_slow",
            "recon_learn_only",
            "recon_slow_no_gain_search",
            "recon_mlp_terminal",
        )

    def _uses_node_params(self) -> bool:
        return self.config.mode in (
            "recon_learn_only",
            "recon_slow_no_gain_search",
            "recon_mlp_terminal",
        )

    def _uses_mlp_terminal(self) -> bool:
        return self.config.mode == "recon_mlp_terminal"

    def _uses_policy_terminal(self) -> bool:
        return self.config.mode == "recon_policy_terminal"

    def learning_mechanisms(self) -> dict[str, bool]:
        return {
            "edge_plasticity": self._uses_fast_plasticity(),
            "bandit_persistence": self._uses_bandit() and not self.config.reset_bandit_each_episode,
            "slow_consolidation": self._uses_slow_consolidation(),
            "node_param_learning": self._uses_node_params(),
            "mlp_terminal": self._uses_mlp_terminal(),
            "policy_terminal": self._uses_policy_terminal(),
            "gain_mutation": self.config.mode
            in ("gain_search_only", "gain_search_recon_fast_bandit"),
        }

    def start_episode(self) -> None:
        self.engine.reset_states()
        reset_episode(self.plasticity_state, self.graph)
        if self.config.reset_bandit_each_episode:
            reset_bandit_episode(self.bandit_state)
        self.last_selected_regime = None
        self.last_fired_edges = []
        self.last_fast_deltas = {}
        self.last_node_param_deltas = {}
        for item in self.node_param_state.values():
            item.reset_episode()
        self.mlp_terminal_state.start_episode(
            self.config.mlp_terminal, self.mlp_rng, self.config.learn and self._uses_mlp_terminal()
        )
        self.policy_terminal_obs_history = []
        self.last_policy_terminal = {}

    def observe_reward(self, reward: float) -> None:
        if not self.config.learn:
            return
        if self.last_selected_regime and self._uses_bandit():
            assign_reward(
                "select_control_regime", self.last_selected_regime, reward, self.bandit_state
            )
        if self._uses_fast_plasticity():
            update_eligibility(
                self.plasticity_state, self.last_fired_edges, self.config.plasticity.lambda_decay
            )
            self.last_fast_deltas = apply_fast_update(
                self.plasticity_state,
                self.graph,
                reward,
                self.last_modulators.eta_tick_eff,
                self.config.plasticity,
            )
        if self._uses_mlp_terminal():
            self.last_mlp_terminal = {
                **self.last_mlp_terminal,
                **self.mlp_terminal_state.update_from_tick(
                    reward, self.config.force_mag, self.config.mlp_terminal
                ),
            }
        if self._uses_node_params():
            self.last_node_param_deltas = update_regime_param_state(
                self.node_param_state,
                self.last_selected_regime,
                reward,
                self.last_proposal.force,
                self.config.force_mag,
                self.config.node_params,
            )

    def act(self, observation: Any, raw_state: Any | None = None) -> tuple[Any, dict[str, Any]]:
        if self.config.mode == "baseline_random":
            action = random_action(self.config.discrete_action_bins)
            bins = max(2, int(self.config.discrete_action_bins))
            force = self.config.force_mag if bins == 2 and action == 1 else -self.config.force_mag
            if bins > 2:
                force = -self.config.force_mag + action * (2.0 * self.config.force_mag / (bins - 1))
            return action_from_force(
                force,
                self.config.action_mode,
                self.config.force_mag,
                self.config.discrete_action_bins,
            ), {"force": force}

        features = features_from_state(observation, raw_state, self.config.n_poles)
        if self.config.mode == "baseline_heuristic":
            force = heuristic_force(features, self.config.force_mag)
            return action_from_force(
                force,
                self.config.action_mode,
                self.config.force_mag,
                self.config.discrete_action_bins,
            ), {"force": force, "features": features}

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
            "suppressed_proposals": [],
        }
        graph_ticks = [self._graph_tick_snapshot(context, [], "root_requested")]
        self.last_fired_edges = []
        for _ in range(self.config.max_engine_ticks):
            now_requested = self.engine.step(context)
            fired_edges = fired_edges_from_requests(self.graph, now_requested)
            self.last_fired_edges.extend(fired_edges)
            graph_ticks.append(self._graph_tick_snapshot(context, fired_edges, "engine_step"))
            if "action" in context:
                break

        action = context.get("action", 0)
        self.last_goal_vector = context.get("goal_vector", {})
        self.last_modulators = compute_modulators(self.last_goal_vector, self.config.modulation)
        self.last_selected_regime = context.get("selected_regime", "recover_worst_pole")
        self.last_proposal = context.get(
            "selected_proposal", ForceProposal("none", 0.0, 0.0, 0.0, "fallback")
        )
        diagnostics = {
            "force": float(context.get("force", 0.0)),
            "goal_vector": self.last_goal_vector,
            "modulators": self.last_modulators.to_dict(),
            "selected_regime": self.last_selected_regime,
            "proposal": asdict(self.last_proposal),
            "proposals": [asdict(item) for item in context.get("proposals", [])],
            "suppressed_proposals": list(context.get("suppressed_proposals", [])),
            "selection_mode": self.config.selection_mode,
            "fired_edges": list(self.last_fired_edges),
            "plasticity": snapshot_plasticity(self.plasticity_state),
            "fast_deltas": dict(self.last_fast_deltas),
            "node_params": snapshot_regime_params(self.node_param_state)
            if self._uses_node_params()
            else {},
            "node_param_deltas": dict(self.last_node_param_deltas),
            "consolidation": self.consolidation.to_dict()
            if self._uses_slow_consolidation()
            else {},
            "consolidation_applied": dict(self.last_consolidation_applied),
            "mlp_terminal": dict(self.last_mlp_terminal),
            "policy_terminal": dict(self.last_policy_terminal),
            "bandit": snapshot_bandit(self.bandit_state),
            "graph_nodes": {nid: node.state.name for nid, node in self.graph.nodes.items()},
            "graph_ticks": graph_ticks,
        }
        return action, diagnostics

    def _graph_tick_snapshot(
        self, context: dict[str, Any], fired_edges: list[dict[str, str]], phase: str
    ) -> dict[str, Any]:
        proposal = context.get("selected_proposal")
        return {
            "engine_tick": int(self.engine.tick),
            "phase": phase,
            "nodes": {nid: node.state.name for nid, node in self.graph.nodes.items()},
            "fired_edges": list(fired_edges),
            "selected_regime": context.get("selected_regime"),
            "force": float(context.get("force", 0.0)),
            "proposal": asdict(proposal) if proposal is not None else None,
            "action_ready": "action" in context,
        }

    def end_episode(
        self, reward_history: list[float], total_return: float, horizon: int
    ) -> dict[str, Any]:
        if not self._uses_slow_consolidation() or not self.config.learn:
            return {}
        edge_delta_sums = {
            key: state.delta_sum
            for key, state in self.plasticity_state.items()
            if abs(state.delta_sum) > 1e-12
        }
        avg_reward_tick = sum(reward_history) / len(reward_history) if reward_history else 0.0
        node_param_delta_sums = {
            regime: dict(item.delta_sum)
            for regime, item in self.node_param_state.items()
            if item.delta_sum
        }
        summary = EpisodeSummary(
            edge_delta_sums=edge_delta_sums,
            avg_reward_tick=avg_reward_tick,
            outcome_score=float(total_return) / max(1, horizon),
            metadata={
                "n_poles": self.config.n_poles,
                "stage": self.config.stage,
                "bandit": snapshot_bandit(self.bandit_state),
                "node_param_delta_sums": node_param_delta_sums,
            },
        )
        self.consolidation.accumulate_episode(summary)
        edge_applied = (
            self.consolidation.apply_to_graph(self.graph)
            if self.consolidation.should_apply()
            else {}
        )
        node_param_applied = {}
        if (
            self.consolidation.episodes_since_apply == 0
            or self.config.node_params.min_episodes <= 1
        ):
            node_param_applied = consolidate_regime_params(
                self.node_param_state,
                summary.outcome_score,
                self.config.node_params,
            )
        mlp_applied = (
            self.mlp_terminal_state.end_episode(total_return, horizon, self.config.mlp_terminal)
            if self._uses_mlp_terminal()
            else {}
        )
        if edge_applied:
            self._sync_plasticity_base_weights()
        applied = {
            "edges": edge_applied,
            "node_params": node_param_applied,
            "mlp_terminal": mlp_applied,
        }
        self.last_consolidation_applied = applied
        return {"summary": summary.__dict__, "applied": applied, "state": self.checkpoint_dict()}

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "n_poles": self.config.n_poles,
                "mode": self.config.mode,
                "stage": self.config.stage,
                "selection_mode": self.config.selection_mode,
                "action_mode": self.config.action_mode,
                "force_mag": self.config.force_mag,
                "discrete_action_bins": self.config.discrete_action_bins,
                "proposal_gains": self.config.proposal_gains.to_dict(),
                "node_params_config": self.config.node_params.__dict__,
                "consolidation_config": self.config.consolidation.__dict__,
                "mlp_terminal_config": self.config.mlp_terminal.__dict__,
                "policy_terminal_path": self.config.policy_terminal_path,
                "policy_terminal_blend": self.config.policy_terminal_blend,
                "policy_terminal_frame_stack": self.config.policy_terminal_frame_stack,
            },
            "edge_weights": {
                f"{edge.src}->{edge.dst}:{edge.ltype.name}": self.edge_weight(
                    edge.src, edge.dst, edge.ltype
                )
                for edge in self.graph.edges
                if edge.ltype in (LinkType.SUB, LinkType.POR)
            },
            "bandit": snapshot_bandit(self.bandit_state),
            "node_params": snapshot_regime_params(self.node_param_state),
            "consolidation": self.consolidation.to_dict(),
            "mlp_terminal": self.mlp_terminal_state.to_dict(),
        }

    def save_checkpoint(self, path: str) -> None:
        import json
        from pathlib import Path

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.checkpoint_dict(), indent=2), encoding="utf-8")

    def save_consolidation_checkpoint(self, path: str) -> None:
        self.save_checkpoint(path)

    def load_consolidation_checkpoint(self, path: str) -> None:
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = data.get("consolidation", data)
        config = ConsolidationConfig(**payload.get("config", {}))
        self.consolidation = ConsolidationEngine(config)
        self.consolidation.total_episodes = int(payload.get("total_episodes", 0))
        self.consolidation.episodes_since_apply = int(payload.get("episodes_since_apply", 0))
        self.consolidation.last_apply_time = payload.get("last_apply_time")
        for key, edge_payload in payload.get("edge_states", {}).items():
            from recon_lite.plasticity import EdgeConsolidationState

            self.consolidation.edge_states[key] = EdgeConsolidationState(**edge_payload)
            self._apply_edge_key_weight(key, float(edge_payload.get("w_base", 1.0)))
        for key, weight in data.get("edge_weights", {}).items():
            self._apply_edge_key_weight(key, float(weight))
        self.node_param_state = {
            regime: RegimeParamState.from_dict(item)
            for regime, item in data.get("node_params", {}).items()
        } or self.node_param_state
        if data.get("mlp_terminal"):
            loaded_mlp = MlpTerminalState.from_dict(data["mlp_terminal"])
            if loaded_mlp.input_size == self.mlp_terminal_state.input_size:
                self.mlp_terminal_state = loaded_mlp
        self._sync_plasticity_base_weights()

    def _sync_plasticity_base_weights(self) -> None:
        for state in self.plasticity_state.values():
            state.w_init = self.edge_weight(state.src, state.dst, state.ltype)

    def _apply_edge_key_weight(self, key: str, weight: float) -> bool:
        src, rest = key.split("->", 1)
        dst, ltype = rest.split(":", 1)
        return self.set_edge_weight(src, dst, ltype, weight)

    def _load_policy_terminal(self, path: str) -> Any:
        try:
            from stable_baselines3 import PPO
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("recon_policy_terminal requires stable-baselines3") from exc
        return PPO.load(path, device="cpu")

    def _force_from_policy_action(self, action: Any) -> float:
        if self.config.action_mode == "continuous":
            return float(
                np.clip(
                    np.asarray(action, dtype=float).reshape(-1)[0],
                    -self.config.force_mag,
                    self.config.force_mag,
                )
            )
        bins = max(2, int(self.config.discrete_action_bins))
        idx = int(np.clip(int(np.asarray(action).reshape(-1)[0]), 0, bins - 1))
        if bins == 2:
            return self.config.force_mag if idx == 1 else -self.config.force_mag
        return float(np.linspace(-self.config.force_mag, self.config.force_mag, bins)[idx])

    def _policy_terminal_observation(self, observation: Any) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        frame_stack = max(1, int(self.config.policy_terminal_frame_stack))
        if frame_stack <= 1:
            return obs
        self.policy_terminal_obs_history.append(obs)
        self.policy_terminal_obs_history = self.policy_terminal_obs_history[-frame_stack:]
        pad_count = frame_stack - len(self.policy_terminal_obs_history)
        frames = [
            self.policy_terminal_obs_history[0]
        ] * pad_count + self.policy_terminal_obs_history
        return np.concatenate(frames).astype(np.float32, copy=False)

    def _policy_terminal_force(self, observation: Any) -> tuple[float | None, dict[str, Any]]:
        if self.policy_terminal_model is None:
            return None, {"available": False, "reason": "no_model"}
        policy_observation = self._policy_terminal_observation(observation)
        action, _state = self.policy_terminal_model.predict(policy_observation, deterministic=True)
        force = self._force_from_policy_action(action)
        return force, {
            "available": True,
            "model_path": self.config.policy_terminal_path,
            "frame_stack": max(1, int(self.config.policy_terminal_frame_stack)),
            "observation_size": int(policy_observation.size),
            "action": np.asarray(action).reshape(-1).tolist(),
            "force": force,
        }

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
        env["features"] = features_from_state(
            env["observation"], env.get("raw_state"), self.config.n_poles
        )
        return True, True

    def _estimate_risk(self, _node, env):
        env["goal_vector"] = compute_cartpole_goal_vector(
            env["features"],
            n_poles=self.config.n_poles,
            stage=self.config.stage,
        )
        env["modulators"] = compute_modulators(env["goal_vector"], self.config.modulation)
        env["selected_regime"] = self._select_regime(
            env["features"], env["goal_vector"], env["modulators"].c_explore_eff
        )
        return True, True

    def edge_weight(self, src: str, dst: str, ltype: LinkType | str = LinkType.SUB) -> float:
        name = ltype.name if isinstance(ltype, LinkType) else str(ltype).upper()
        for edge in self.graph.edges:
            if edge.src == src and edge.dst == dst and edge.ltype.name == name:
                try:
                    return float(edge.w[0]) if hasattr(edge.w, "__len__") else float(edge.w)
                except Exception:
                    return 1.0
        return 1.0

    def set_edge_weight(self, src: str, dst: str, ltype: LinkType | str, weight: float) -> bool:
        name = ltype.name if isinstance(ltype, LinkType) else str(ltype).upper()
        for edge in self.graph.edges:
            if edge.src == src and edge.dst == dst and edge.ltype.name == name:
                edge.w = float(weight)
                return True
        return False

    def _bandit_multiplier(self, regime: str, c_explore_eff: float = 0.0) -> float:
        if not self._uses_bandit():
            return 1.0
        arm = self.bandit_state.get("select_control_regime", {}).get(regime)
        if arm is None or arm.pulls == 0:
            return 1.0
        total_pulls = sum(
            item.pulls for item in self.bandit_state.get("select_control_regime", {}).values()
        )
        score = ucb_score(arm, total_pulls, c_explore_eff or self.config.bandit.c_explore)
        return max(0.05, min(5.0, 1.0 + score))

    def _score_proposal(
        self, proposal: ForceProposal, selected: str | None, c_explore_eff: float
    ) -> ForceProposal:
        regime = proposal.source_node
        is_selected = regime == selected
        selection_mode = self.config.selection_mode
        select_weight = self.edge_weight("select_control_regime", regime, LinkType.SUB)
        proposal_weight = self.edge_weight(regime, f"{regime}_proposal", LinkType.SUB)
        bandit_score = self._bandit_multiplier(regime, c_explore_eff)
        if selection_mode == "hard_select" and not is_selected:
            proposal.confidence = 0.0
            proposal.urgency = 0.0
            proposal.score = 0.0
            proposal.select_edge_weight = select_weight
            proposal.proposal_edge_weight = proposal_weight
            proposal.bandit_score = bandit_score
            proposal.selection_multiplier = 0.0
            proposal.selected = False
            proposal.suppressed = True
            proposal.selection_mode = selection_mode
            return proposal
        selection_multiplier = 1.0 if is_selected else 0.35
        base_priority = max(0.01, proposal.raw_confidence) * (1.0 + proposal.raw_urgency)
        weighted_score = (
            base_priority * select_weight * proposal_weight * bandit_score * selection_multiplier
        )
        proposal.confidence = max(
            0.0,
            proposal.raw_confidence
            * select_weight
            * proposal_weight
            * bandit_score
            * selection_multiplier,
        )
        proposal.urgency = max(0.0, proposal.raw_urgency * proposal_weight)
        proposal.score = weighted_score
        proposal.select_edge_weight = select_weight
        proposal.proposal_edge_weight = proposal_weight
        proposal.bandit_score = bandit_score
        proposal.selection_multiplier = selection_multiplier
        proposal.selected = is_selected
        proposal.suppressed = False
        proposal.selection_mode = selection_mode
        return proposal

    def _proposal(self, node, env):
        regime = node.meta["regime"]
        selected = env.get("selected_regime")
        proposal = propose_force_for_regime(
            regime,
            env["features"],
            self.config.force_mag,
            self.config.proposal_gains,
        )
        if self._uses_node_params():
            proposal = apply_node_params(
                proposal,
                self.node_param_state[regime].current,
                env["features"],
                self.config.force_mag,
            )
        if self._uses_mlp_terminal() and regime == "stabilize_chain":
            correction, mlp_info = self.mlp_terminal_state.force(
                env["features"], self.config.force_mag
            )
            blend = max(0.0, min(1.0, self.config.mlp_terminal.blend))
            base_force = proposal.force
            proposal.force = max(
                -self.config.force_mag, min(self.config.force_mag, base_force + blend * correction)
            )
            self.mlp_terminal_state.last_force = proposal.force
            proposal.reason = f"{proposal.reason}; mlp_terminal"
            mlp_info["blend"] = blend
            mlp_info["base_force"] = base_force
            mlp_info["corrected_force"] = proposal.force
            self.last_mlp_terminal = mlp_info
        if self._uses_policy_terminal() and regime == "stabilize_chain":
            policy_force, policy_info = self._policy_terminal_force(env["observation"])
            if policy_force is not None:
                base_force = proposal.force
                blend = max(0.0, min(1.0, self.config.policy_terminal_blend))
                proposal.force = float(
                    np.clip(
                        base_force + blend * (policy_force - base_force),
                        -self.config.force_mag,
                        self.config.force_mag,
                    )
                )
                proposal.confidence = max(proposal.confidence, 0.9)
                proposal.reason = f"{proposal.reason}; policy_terminal"
                policy_info["blend"] = blend
                policy_info["base_force"] = base_force
                policy_info["policy_force"] = policy_force
                policy_info["proposal_force"] = proposal.force
            self.last_policy_terminal = policy_info
        proposal.raw_confidence = proposal.confidence
        proposal.raw_urgency = proposal.urgency
        proposal = self._score_proposal(
            proposal, selected, env.get("modulators", self.last_modulators).c_explore_eff
        )
        if proposal.suppressed:
            env.setdefault("suppressed_proposals", []).append(asdict(proposal))
        else:
            env.setdefault("proposals", []).append(proposal)
        return True, True

    def _arbitrate_force(self, _node, env):
        proposal = arbitrate_force(env.get("proposals", []), self.config.force_mag)
        env["selected_proposal"] = proposal
        env["force"] = proposal.force
        return True, True

    def _apply_force(self, _node, env):
        env["action"] = action_from_force(
            float(env.get("force", 0.0)),
            self.config.action_mode,
            self.config.force_mag,
            self.config.discrete_action_bins,
        )
        return True, True

    def _pole_sensor(self, node, env):
        idx = int(node.meta["pole_index"])
        env.setdefault("pole_sensor_values", {})[idx] = env["features"].poles[idx]
        return True, True

    def _select_regime(
        self, features: StateFeatures, goal_vector: dict[str, Any], c_explore_eff: float
    ) -> str:
        if self._uses_bandit():
            arms = self.bandit_state.get("select_control_regime", {})
            has_priors = any(arm.pulls > 0 for arm in arms.values())
            if self.config.learn or has_priors:
                child = choose_child(
                    "select_control_regime", self.bandit_state, c_explore_eff, self.config.bandit
                )
                if child:
                    return child
        if abs(features.x) > 1.5:
            return "avoid_rail"
        if goal_vector.get("max_velocity_pressure", 0.0) > 0.65:
            return "damp_energy"
        if self.config.n_poles > 1 and features.worst_pole_index > 0:
            return "stabilize_chain"
        return "recover_worst_pole"
