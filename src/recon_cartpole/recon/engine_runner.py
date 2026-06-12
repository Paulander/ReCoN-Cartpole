from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
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
from recon_cartpole.control.policy_observation import policy_observation_from_state
from recon_cartpole.control.residual_features import residual_aux_features
from recon_cartpole.control.scripts import (
    REGIMES,
    ForceProposal,
    ProposalGains,
    propose_force_for_regime,
)
from recon_cartpole.control.sensors import StateFeatures, features_from_state

from .fired_edges import fired_edges_from_requests
from .graph_factory import GraphConfig, build_cartpole_graph, trainable_edge_whitelist
from .mingru_terminal import MinGRUPrediction, MinGRUTerminal, MinGRUTerminalConfig
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




class TorchResidualPolicy:
    def __init__(self, path: str):
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("torch residual policy terminals require torch") from exc
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = dict(payload.get("meta", {}))
        input_size = int(meta.get("input_size", 0))
        hidden_size = int(meta.get("hidden_size", 64))
        classes = int(meta.get("classes", 0))
        if input_size <= 0 or classes <= 0:
            raise ValueError(f"invalid torch residual terminal metadata in {path}")
        self.torch = torch
        self.meta = meta
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, classes),
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def predict(self, observation: Any, deterministic: bool = True):
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        expected = int(self.meta["input_size"])
        if obs.shape[1] != expected:
            raise ValueError(f"torch residual terminal observation size mismatch: obs={obs.shape[1]} expected={expected}")
        with self.torch.no_grad():
            logits = self.model(self.torch.as_tensor(obs, dtype=self.torch.float32))
            action = int(self.torch.argmax(logits, dim=1).cpu().numpy()[0])
        return action, None


@dataclass
class Pole1FixConfig:
    enabled: bool = False
    angle_threshold: float = 0.14
    velocity_threshold: float = 1.2
    urgency_boost: float = 0.45
    confidence_boost: float = 0.20
    force_blend: float = 0.35
    rail_guard: float = 2.05
    velocity_mix: float = 0.30


@dataclass
class RescueConfig:
    enabled: bool = False
    late_episode_conservative_mode: bool = False
    late_start_step: int = 400
    late_rail_guard: float = 2.05
    rail_vs_pole_priority_gate: bool = False
    rail_imminent_x: float = 2.10
    terminal_force_passthrough_high_confidence: bool = False
    passthrough_start_step: int = 400
    passthrough_angle_threshold: float = 0.14
    passthrough_velocity_pressure: float = 0.65
    anti_oscillation_damper: bool = False
    oscillation_window: int = 12
    oscillation_flip_threshold: int = 8
    pole1_emergency_guard_v2: bool = False
    pole1_angle_threshold: float = 0.18
    pole1_velocity_threshold: float = 0.75
    pole1_velocity_mix: float = 0.25
    pole1_force_blend: float = 0.75


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
    policy_terminal_scope: str = "stabilize_chain"
    policy_terminal_observation_mode: str = "env"
    policy_terminal_recurrent: bool = False
    policy_terminal_normalizer_path: str = ""
    residual_policy_terminal_path: str = ""
    residual_policy_terminal_mode: str = "force"
    residual_policy_terminal_action_bins: int = 5
    residual_policy_terminal_max_force: float = 4.0
    residual_policy_terminal_gate_threshold: float = 0.30
    residual_policy_terminal_feature_mode: str = "basic"
    residual_policy_terminal_hold_steps: int = 1
    mingru_terminal: MinGRUTerminalConfig = field(default_factory=MinGRUTerminalConfig)
    pole1_fix: Pole1FixConfig = field(default_factory=Pole1FixConfig)
    rescue: RescueConfig = field(default_factory=RescueConfig)


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
        if self._uses_pole1_fix() and not self.config.pole1_fix.enabled:
            self.config.pole1_fix.enabled = True
        self.mlp_terminal_state = MlpTerminalState.create(
            self.config.n_poles, self.config.mlp_terminal.hidden_size
        )
        self.mlp_rng = np.random.default_rng(1009 + self.config.n_poles)
        self.last_mlp_terminal: dict[str, Any] = {}
        self.mingru_terminal: MinGRUTerminal | None = None
        self.last_mingru_terminal: dict[str, Any] = {}
        if self._uses_mingru_terminal() and not self.config.mingru_terminal.enabled:
            self.config.mingru_terminal.enabled = True
        if self._uses_mingru_terminal():
            self.mingru_terminal = MinGRUTerminal(
                self.config.n_poles,
                self.config.force_mag,
                self.config.discrete_action_bins,
                self.config.mingru_terminal,
            )
        self.policy_terminal_model: Any | None = None
        self.policy_terminal_obs_history: list[np.ndarray] = []
        self.policy_terminal_state: Any | None = None
        self.policy_terminal_episode_start = np.ones((1,), dtype=bool)
        self.last_policy_terminal: dict[str, Any] = {}
        self.policy_terminal_normalizer: dict[str, np.ndarray] | None = self._load_policy_terminal_normalizer(
            self.config.policy_terminal_normalizer_path
        )
        if self._uses_policy_terminal() and self.config.policy_terminal_path:
            self.policy_terminal_model = self._load_policy_terminal(
                self.config.policy_terminal_path
            )
        self.residual_policy_terminal_model: Any | None = None
        self.last_residual_policy_terminal: dict[str, Any] = {}
        self.residual_option_shift = 0
        self.residual_option_remaining = 0
        if self.config.residual_policy_terminal_path:
            self.residual_policy_terminal_model = self._load_feedforward_policy_terminal(
                self.config.residual_policy_terminal_path
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
        self.episode_step = 0
        self.recent_forces: list[float] = []
        self.residual_option_shift = 0
        self.residual_option_remaining = 0
        self.last_rescue: dict[str, Any] = {}

    def _uses_fast_plasticity(self) -> bool:
        return self.config.mode in (
            "recon_fast",
            "recon_fast_bandit",
            "recon_slow",
            "gain_search_recon_fast_bandit",
            "recon_learn_only",
            "recon_slow_no_gain_search",
            "recon_mlp_terminal",
            "recon_mingru_terminal_plus_recon_learning",
            "recon_feedforward_terminal_plus_recon_learning",
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
            "recon_mingru_terminal_plus_recon_learning",
            "recon_feedforward_terminal_plus_recon_learning",
        )

    def _uses_slow_consolidation(self) -> bool:
        return self.config.mode in (
            "recon_slow",
            "recon_learn_only",
            "recon_slow_no_gain_search",
            "recon_mlp_terminal",
            "recon_mingru_terminal_plus_recon_learning",
            "recon_feedforward_terminal_plus_recon_learning",
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
        return self.config.mode in (
            "recon_policy_terminal",
            "recon_recurrent_policy_terminal",
            "recon_feedforward_terminal_frozen",
            "recon_feedforward_terminal_plus_recon_learning",
            "recon_feedforward_terminal_with_pole1_fix",
        )

    def _uses_mingru_terminal(self) -> bool:
        return self.config.mode in (
            "recon_mingru_terminal",
            "recon_mingru_terminal_plus_recon_learning",
        )

    def _uses_pole1_fix(self) -> bool:
        return self.config.mode == "recon_feedforward_terminal_with_pole1_fix"

    def learning_mechanisms(self) -> dict[str, bool]:
        return {
            "edge_plasticity": self._uses_fast_plasticity(),
            "bandit_persistence": self._uses_bandit() and not self.config.reset_bandit_each_episode,
            "slow_consolidation": self._uses_slow_consolidation(),
            "node_param_learning": self._uses_node_params(),
            "mlp_terminal": self._uses_mlp_terminal(),
            "policy_terminal": self._uses_policy_terminal(),
            "recurrent_policy_terminal": self._uses_policy_terminal()
            and self.config.policy_terminal_recurrent,
            "minGRU_terminal": self._uses_mingru_terminal(),
            "pole1_fix": self.config.pole1_fix.enabled or self._uses_pole1_fix(),
            "rescue_patches": self.config.rescue.enabled,
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
        self.policy_terminal_state = None
        self.policy_terminal_episode_start = np.ones((1,), dtype=bool)
        self.last_policy_terminal = {}
        self.last_mingru_terminal = {}
        self.last_rescue = {}
        self.episode_step = 0
        self.recent_forces = []
        if self.mingru_terminal is not None:
            self.mingru_terminal.reset()

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
            "episode_step": self.episode_step,
            "rescue": {},
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
            "mingru_terminal": dict(self.last_mingru_terminal),
            "mingru_passthrough": dict(context.get("mingru_passthrough", {})),
            "bandit": snapshot_bandit(self.bandit_state),
            "graph_nodes": {nid: node.state.name for nid, node in self.graph.nodes.items()},
            "graph_ticks": graph_ticks,
            "rescue": dict(context.get("rescue", {})),
        }
        self.last_rescue = dict(context.get("rescue", {}))
        self.recent_forces.append(float(context.get("force", 0.0)))
        self.recent_forces = self.recent_forces[-max(2, int(self.config.rescue.oscillation_window)) :]
        self.episode_step += 1
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
                "policy_terminal_scope": self.config.policy_terminal_scope,
                "policy_terminal_observation_mode": self.config.policy_terminal_observation_mode,
                "policy_terminal_recurrent": self.config.policy_terminal_recurrent,
                "policy_terminal_normalizer_path": self.config.policy_terminal_normalizer_path,
                "residual_policy_terminal_path": self.config.residual_policy_terminal_path,
                "residual_policy_terminal_mode": self.config.residual_policy_terminal_mode,
                "residual_policy_terminal_feature_mode": self.config.residual_policy_terminal_feature_mode,
                "mingru_terminal_config": self.config.mingru_terminal.__dict__,
                "pole1_fix_config": self.config.pole1_fix.__dict__,
                "rescue_config": self.config.rescue.__dict__,
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


    def _load_policy_terminal_normalizer(self, path: str) -> dict[str, np.ndarray] | None:
        if not path:
            return None
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        mean = np.asarray(data.get("mean", []), dtype=np.float32).reshape(-1)
        var = np.asarray(data.get("var", []), dtype=np.float32).reshape(-1)
        if mean.size == 0 or var.size == 0 or mean.shape != var.shape:
            raise ValueError(f"invalid policy terminal normalizer at {path}")
        return {
            "mean": mean,
            "var": np.maximum(var, 1e-12),
            "epsilon": np.asarray([float(data.get("epsilon", 1e-8))], dtype=np.float32),
            "clip_obs": np.asarray([float(data.get("clip_obs", 10.0))], dtype=np.float32),
        }

    def _normalize_policy_terminal_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        normalizer = self.policy_terminal_normalizer
        if normalizer is None:
            return obs
        mean = normalizer["mean"]
        var = normalizer["var"]
        if obs.shape != mean.shape:
            raise ValueError(
                "policy terminal normalizer shape mismatch: "
                f"obs={obs.shape} mean={mean.shape}"
            )
        epsilon = float(normalizer["epsilon"][0])
        clip_obs = float(normalizer["clip_obs"][0])
        normalized = (obs - mean) / np.sqrt(var + epsilon)
        return np.clip(normalized, -clip_obs, clip_obs).astype(np.float32, copy=False)

    def _load_feedforward_policy_terminal(self, path: str) -> Any:
        if str(path).endswith(".pt"):
            return TorchResidualPolicy(path)
        try:
            from stable_baselines3 import PPO
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("residual policy terminals require stable-baselines3") from exc
        return PPO.load(path, device="cpu")

    def _residual_risk_gate(self, raw_state: Any | None) -> float:
        raw = np.asarray(raw_state, dtype=np.float32).reshape(-1) if raw_state is not None else np.asarray([], dtype=np.float32)
        n = int(self.config.n_poles)
        if raw.size < 2 + 2 * n:
            return 0.0
        x = abs(float(raw[0])) / 2.4
        theta = float(np.max(np.abs(raw[2 : 2 + n]))) / 0.20943951023931953
        theta_dot = float(np.max(np.abs(raw[2 + n : 2 + 2 * n]))) / 5.0
        late = self.episode_step / 500.0
        return float(np.clip(max(x, theta, 0.5 * theta_dot, late if late > 0.75 else 0.0), 0.0, 1.0))

    def _force_to_discrete_index(self, force: float) -> int:
        bins = max(2, int(self.config.discrete_action_bins))
        if bins == 2:
            return 1 if force >= 0.0 else 0
        values = np.linspace(-self.config.force_mag, self.config.force_mag, bins)
        return int(np.argmin(np.abs(values - float(force))))

    def _apply_residual_policy_terminal(
        self,
        policy_observation: np.ndarray,
        base_force: float,
        raw_state: Any | None,
    ) -> tuple[float, dict[str, Any]]:
        if self.residual_policy_terminal_model is None:
            return base_force, {"available": False, "reason": "no_model"}
        gate = self._residual_risk_gate(raw_state)
        aux_features = residual_aux_features(
            raw_state,
            n_poles=self.config.n_poles,
            force_mag=self.config.force_mag,
            base_force=base_force,
            previous_force=self.recent_forces[-1] if self.recent_forces else 0.0,
            horizon=500,
            episode_step=self.episode_step,
            mode=self.config.residual_policy_terminal_feature_mode,
            proposal_gains=self.config.proposal_gains,
        )
        residual_obs = np.concatenate(
            [
                np.asarray(policy_observation, dtype=np.float32).reshape(-1),
                aux_features,
            ]
        ).astype(np.float32, copy=False)
        action, _state = self.residual_policy_terminal_model.predict(residual_obs, deterministic=True)
        action_idx = int(
            np.clip(
                int(np.asarray(action).reshape(-1)[0]),
                0,
                max(2, int(self.config.residual_policy_terminal_action_bins)) - 1,
            )
        )
        mode = self.config.residual_policy_terminal_mode
        final_force = float(base_force)
        delta = 0.0
        requested_shift = 0
        applied_shift = 0
        option_reused = False
        hold_steps = max(1, int(self.config.residual_policy_terminal_hold_steps))
        if mode == "bin_delta" and self.config.action_mode == "discrete":
            action_bins = max(2, int(self.config.residual_policy_terminal_action_bins))
            max_shift = action_bins // 2
            requested_shift = action_idx - max_shift
            if self.residual_option_remaining > 0 and self.residual_option_shift != 0:
                applied_shift = int(self.residual_option_shift)
                self.residual_option_remaining -= 1
                option_reused = True
            elif gate >= float(self.config.residual_policy_terminal_gate_threshold):
                applied_shift = requested_shift
                if applied_shift != 0 and hold_steps > 1:
                    self.residual_option_shift = int(applied_shift)
                    self.residual_option_remaining = hold_steps - 1
                else:
                    self.residual_option_shift = 0
                    self.residual_option_remaining = 0
            else:
                self.residual_option_shift = 0
                self.residual_option_remaining = 0
            base_idx = self._force_to_discrete_index(base_force)
            final_idx = int(
                np.clip(base_idx + applied_shift, 0, int(self.config.discrete_action_bins) - 1)
            )
            values = np.linspace(
                -self.config.force_mag, self.config.force_mag, int(self.config.discrete_action_bins)
            )
            final_force = float(values[final_idx])
            delta = final_force - float(base_force)
        else:
            values = np.linspace(
                -self.config.residual_policy_terminal_max_force,
                self.config.residual_policy_terminal_max_force,
                max(2, int(self.config.residual_policy_terminal_action_bins)),
            )
            requested_delta = float(values[action_idx])
            delta = gate * requested_delta
            final_force = float(
                np.clip(base_force + delta, -self.config.force_mag, self.config.force_mag)
            )
        info = {
            "available": True,
            "model_path": self.config.residual_policy_terminal_path,
            "mode": mode,
            "feature_mode": self.config.residual_policy_terminal_feature_mode,
            "aux_feature_size": int(aux_features.size),
            "observation_size": int(residual_obs.size),
            "action": [action_idx],
            "risk_gate": gate,
            "requested_shift": int(requested_shift),
            "applied_shift": int(applied_shift),
            "hold_steps": int(hold_steps),
            "option_reused": bool(option_reused),
            "option_remaining": int(self.residual_option_remaining),
            "base_force": float(base_force),
            "residual_delta": float(delta),
            "force": float(final_force),
        }
        return final_force, info

    def _load_policy_terminal(self, path: str) -> Any:
        if self.config.policy_terminal_recurrent:
            try:
                from sb3_contrib import RecurrentPPO
            except Exception as exc:  # pragma: no cover - optional dependency path
                raise RuntimeError("recurrent policy terminals require sb3-contrib") from exc
            return RecurrentPPO.load(path, device="cpu")
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

    def _policy_terminal_observation(
        self, observation: Any, raw_state: Any | None = None
    ) -> np.ndarray:
        obs = policy_observation_from_state(
            observation,
            raw_state,
            self.config.n_poles,
            self.config.policy_terminal_observation_mode,
            previous_force=self.recent_forces[-1] if self.recent_forces else 0.0,
            force_mag=self.config.force_mag,
        )
        frame_stack = max(1, int(self.config.policy_terminal_frame_stack))
        if frame_stack <= 1:
            return self._normalize_policy_terminal_observation(obs)
        self.policy_terminal_obs_history.append(obs)
        self.policy_terminal_obs_history = self.policy_terminal_obs_history[-frame_stack:]
        pad_count = frame_stack - len(self.policy_terminal_obs_history)
        frames = [
            self.policy_terminal_obs_history[0]
        ] * pad_count + self.policy_terminal_obs_history
        stacked = np.concatenate(frames).astype(np.float32, copy=False)
        return self._normalize_policy_terminal_observation(stacked)

    def _policy_terminal_applies(self, regime: str, selected: str | None) -> bool:
        scope = self.config.policy_terminal_scope
        if scope == "all":
            return True
        if scope == "selected":
            return regime == selected
        return regime == "stabilize_chain"

    def _policy_terminal_force(
        self, observation: Any, raw_state: Any | None = None, context: dict[str, Any] | None = None
    ) -> tuple[float | None, dict[str, Any]]:
        if self.policy_terminal_model is None:
            return None, {"available": False, "reason": "no_model"}
        if context is not None and "_policy_terminal_cache" in context:
            cached = context["_policy_terminal_cache"]
            return cached["force"], dict(cached["info"])
        policy_observation = self._policy_terminal_observation(observation, raw_state)
        recurrent_episode_start = bool(self.policy_terminal_episode_start[0])
        if self.config.policy_terminal_recurrent:
            action, self.policy_terminal_state = self.policy_terminal_model.predict(
                policy_observation,
                state=self.policy_terminal_state,
                episode_start=self.policy_terminal_episode_start,
                deterministic=True,
            )
            self.policy_terminal_episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _state = self.policy_terminal_model.predict(
                policy_observation, deterministic=True
            )
        base_force = self._force_from_policy_action(action)
        force = base_force
        residual_info = {"available": False, "reason": "disabled"}
        if self.residual_policy_terminal_model is not None:
            force, residual_info = self._apply_residual_policy_terminal(
                policy_observation, base_force, raw_state
            )
        self.last_residual_policy_terminal = dict(residual_info)
        info = {
            "available": True,
            "model_path": self.config.policy_terminal_path,
            "frame_stack": max(1, int(self.config.policy_terminal_frame_stack)),
            "observation_size": int(policy_observation.size),
            "scope": self.config.policy_terminal_scope,
            "observation_mode": self.config.policy_terminal_observation_mode,
            "normalizer_path": self.config.policy_terminal_normalizer_path,
            "normalizer_applied": self.policy_terminal_normalizer is not None,
            "recurrent": bool(self.config.policy_terminal_recurrent),
            "episode_start": recurrent_episode_start
            if self.config.policy_terminal_recurrent
            else False,
            "action": np.asarray(action).reshape(-1).tolist(),
            "base_force": base_force,
            "force": force,
            "residual_policy_terminal": residual_info,
        }
        if context is not None:
            context["_policy_terminal_cache"] = {"force": force, "info": dict(info)}
        return force, info

    def _mingru_terminal_applies(self, regime: str, selected: str | None) -> bool:
        scope = self.config.mingru_terminal.scope
        if scope == "all":
            return True
        if scope == "selected":
            return regime == selected
        return regime == "stabilize_chain"

    def _mingru_terminal_force(
        self, observation: Any, raw_state: Any | None = None, context: dict[str, Any] | None = None
    ) -> tuple[MinGRUPrediction, dict[str, Any]]:
        if self.mingru_terminal is None:
            pred = MinGRUPrediction(None, 0.0, 0.0, 0.0, 0.0, 0, valid=False, reason="no_model")
            return pred, {"available": False, "reason": "no_model"}
        if context is not None and "_mingru_terminal_cache" in context:
            cached = context["_mingru_terminal_cache"]
            return cached["prediction"], dict(cached["info"])
        try:
            prediction = self.mingru_terminal.predict(observation, raw_state, context)
        except Exception as exc:  # pragma: no cover - defensive optional model path
            prediction = MinGRUPrediction(None, 0.0, 0.0, 1.0, 0.0, 0, valid=False, reason=type(exc).__name__)
        info = {
            "available": bool(prediction.valid and prediction.force is not None),
            "model_path": self.config.mingru_terminal.checkpoint_path,
            "checkpoint_path": self.config.mingru_terminal.checkpoint_path,
            "loaded_checkpoint": getattr(self.mingru_terminal, "loaded_checkpoint", ""),
            "scope": self.config.mingru_terminal.scope,
            "observation_mode": self.config.mingru_terminal.observation_mode,
            "sequence_length": int(prediction.sequence_length),
            "configured_sequence_length": max(1, int(self.config.mingru_terminal.sequence_length)),
            "force": prediction.force,
            "confidence": float(prediction.confidence),
            "value": float(prediction.value),
            "failure_probability": float(prediction.failure_probability),
            "hidden_norm": float(prediction.hidden_norm),
            "logits": list(prediction.logits),
            "valid": bool(prediction.valid),
            "reason": prediction.reason,
        }
        if context is not None:
            context["_mingru_terminal_cache"] = {"prediction": prediction, "info": dict(info)}
        return prediction, info

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

    def _rescue_enabled(self) -> bool:
        return bool(self.config.rescue.enabled)

    def _rail_imminent(self, features: StateFeatures) -> bool:
        cfg = self.config.rescue
        if abs(features.x) >= cfg.rail_imminent_x:
            return True
        return bool(abs(features.x) > 1.5 and features.x * features.x_dot > 0.0)

    def _recent_force_flips(self) -> int:
        forces = [force for force in self.recent_forces if abs(force) > 1e-6]
        return sum(1 for a, b in zip(forces, forces[1:]) if np.sign(a) != np.sign(b))

    def _high_risk_for_passthrough(self, env: dict[str, Any]) -> bool:
        cfg = self.config.rescue
        features = env["features"]
        if self.episode_step < cfg.passthrough_start_step:
            return False
        if self._rail_imminent(features):
            return False
        if features.max_angle_abs >= cfg.passthrough_angle_threshold:
            return True
        if env.get("goal_vector", {}).get("max_velocity_pressure", 0.0) >= cfg.passthrough_velocity_pressure:
            return True
        if cfg.anti_oscillation_damper and self._recent_force_flips() >= cfg.oscillation_flip_threshold:
            return True
        return False

    def _rescue_terminal_applies(self, regime: str, selected: str | None, env: dict[str, Any]) -> bool:
        cfg = self.config.rescue
        if not (self._rescue_enabled() and cfg.terminal_force_passthrough_high_confidence):
            return False
        if regime != selected:
            return False
        if regime == "avoid_rail" and self._rail_imminent(env["features"]):
            return False
        return self._high_risk_for_passthrough(env)

    def _apply_pole1_emergency_guard(self, proposal: ForceProposal, env: dict[str, Any]) -> ForceProposal:
        cfg = self.config.rescue
        if not (self._rescue_enabled() and cfg.pole1_emergency_guard_v2):
            return proposal
        if self.config.n_poles < 2 or len(env["features"].poles) < 2:
            return proposal
        if self._rail_imminent(env["features"]):
            return proposal
        pole = env["features"].poles[1]
        moving_outward = abs(pole.theta) > cfg.pole1_angle_threshold and pole.theta * pole.theta_dot > 0.0
        fast_outward = abs(pole.theta_dot) > cfg.pole1_velocity_threshold and pole.theta * pole.theta_dot > 0.0
        if not (moving_outward or fast_outward):
            return proposal
        desired_force = self.config.force_mag if pole.theta + cfg.pole1_velocity_mix * pole.theta_dot >= 0.0 else -self.config.force_mag
        base_force = proposal.force
        blend = max(0.0, min(1.0, cfg.pole1_force_blend))
        proposal.force = float(np.clip(base_force + blend * (desired_force - base_force), -self.config.force_mag, self.config.force_mag))
        proposal.confidence = max(proposal.confidence, 0.95)
        proposal.urgency = max(proposal.urgency, 1.0)
        proposal.reason = f"{proposal.reason}; pole1_emergency_v2 desired={desired_force:.2f} base={base_force:.2f}"
        env.setdefault("rescue", {}).setdefault("events", []).append("pole1_emergency_guard_v2")
        return proposal


    def _apply_pole1_fix(self, proposal: ForceProposal, env: dict[str, Any]) -> ForceProposal:
        cfg = self.config.pole1_fix
        if not cfg.enabled or self.config.n_poles < 2 or len(env["features"].poles) < 2:
            return proposal
        if proposal.source_node not in ("stabilize_chain", "recover_worst_pole", "damp_energy"):
            return proposal
        features = env["features"]
        pole = features.poles[1]
        pressure = max(
            abs(pole.theta) / max(cfg.angle_threshold, 1e-9),
            abs(pole.theta_dot) / max(cfg.velocity_threshold, 1e-9),
        )
        if pressure < 1.0:
            return proposal
        if abs(features.x) >= cfg.rail_guard:
            proposal.reason = f"{proposal.reason}; pole1_fix_rail_guard"
            return proposal
        desired_force = (
            self.config.force_mag
            if pole.theta + cfg.velocity_mix * pole.theta_dot >= 0.0
            else -self.config.force_mag
        )
        sign_mismatch = (
            abs(proposal.force) > 1e-9 and np.sign(proposal.force) != np.sign(desired_force)
        )
        blend = max(0.0, min(1.0, cfg.force_blend if sign_mismatch else cfg.force_blend * 0.35))
        base_force = proposal.force
        proposal.force = float(
            np.clip(
                base_force + blend * (desired_force - base_force),
                -self.config.force_mag,
                self.config.force_mag,
            )
        )
        proposal.confidence = min(1.0, proposal.confidence + cfg.confidence_boost)
        proposal.urgency = min(1.5, proposal.urgency + cfg.urgency_boost * min(2.0, pressure))
        proposal.reason = (
            f"{proposal.reason}; pole1_fix pressure={pressure:.2f} "
            f"desired={desired_force:.2f} base={base_force:.2f}"
        )
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
        policy_applies = self._uses_policy_terminal() and (
            self._policy_terminal_applies(regime, selected)
            or self._rescue_terminal_applies(regime, selected, env)
        )
        if policy_applies:
            policy_force, policy_info = self._policy_terminal_force(
                env["observation"], env.get("raw_state"), env
            )
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
                policy_info["applied_regime"] = regime
                policy_info["rescue_passthrough"] = bool(self._rescue_terminal_applies(regime, selected, env))
                if policy_info["rescue_passthrough"]:
                    rescue_info = env.setdefault("rescue", {})
                    rescue_info.setdefault("events", []).append("terminal_force_passthrough_high_confidence")
                    rescue_info["policy_terminal"] = dict(policy_info)
                    proposal.reason = f"{proposal.reason}; rescue_passthrough"
                policy_info["applied_regimes"] = sorted(
                    set(env.setdefault("_policy_terminal_applied_regimes", []) + [regime])
                )
                env["_policy_terminal_applied_regimes"] = policy_info["applied_regimes"]
            self.last_policy_terminal = policy_info
        if self._uses_mingru_terminal() and self._mingru_terminal_applies(regime, selected):
            prediction, mingru_info = self._mingru_terminal_force(
                env["observation"], env.get("raw_state"), env
            )
            base_force = proposal.force
            confidence = max(0.0, min(1.0, float(prediction.confidence)))
            blend = max(0.0, min(1.0, self.config.mingru_terminal.blend))
            confidence_floor = max(0.0, min(1.0, self.config.mingru_terminal.confidence_floor))
            applied = bool(prediction.valid and prediction.force is not None and confidence >= confidence_floor)
            if applied:
                terminal_force = float(prediction.force)
                proposal.force = float(
                    np.clip(
                        base_force + blend * (terminal_force - base_force),
                        -self.config.force_mag,
                        self.config.force_mag,
                    )
                )
                proposal.reason = f"{proposal.reason}; mingru_terminal"
            proposal.confidence = max(0.0, proposal.confidence * max(confidence, confidence_floor))
            mingru_info["blend"] = blend
            mingru_info["base_force"] = base_force
            mingru_info["terminal_force"] = prediction.force
            mingru_info["proposal_force"] = proposal.force
            mingru_info["confidence_floor"] = confidence_floor
            mingru_info["applied"] = applied
            mingru_info["applied_regime"] = regime
            mingru_info["applied_regimes"] = sorted(
                set(env.setdefault("_mingru_terminal_applied_regimes", []) + [regime])
            )
            env["_mingru_terminal_applied_regimes"] = mingru_info["applied_regimes"]
            self.last_mingru_terminal = mingru_info
        proposal = self._apply_pole1_emergency_guard(proposal, env)
        proposal = self._apply_pole1_fix(proposal, env)
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
        proposal = self._maybe_apply_mingru_passthrough(proposal, env)
        env["selected_proposal"] = proposal
        env["force"] = proposal.force
        return True, True

    @staticmethod
    def _prediction_logit_margin(prediction: MinGRUPrediction) -> float:
        logits = [float(value) for value in prediction.logits if np.isfinite(float(value))]
        if len(logits) < 2:
            return 0.0
        top_two = sorted(logits, reverse=True)[:2]
        return float(top_two[0] - top_two[1])

    def _maybe_apply_mingru_passthrough(self, proposal: ForceProposal, env: dict[str, Any]) -> ForceProposal:
        cfg = self.config.mingru_terminal
        if not (self._uses_mingru_terminal() and cfg.passthrough_enabled):
            return proposal
        prediction, mingru_info = self._mingru_terminal_force(
            env["observation"], env.get("raw_state"), env
        )
        confidence = max(0.0, min(1.0, float(prediction.confidence)))
        floor = max(0.0, min(1.0, float(cfg.passthrough_confidence_floor)))
        margin = self._prediction_logit_margin(prediction)
        margin_floor = max(0.0, float(cfg.passthrough_logit_margin_floor))
        applied = bool(
            prediction.valid
            and prediction.force is not None
            and confidence >= floor
            and margin >= margin_floor
        )
        mingru_info["passthrough_enabled"] = True
        mingru_info["passthrough_confidence_floor"] = floor
        mingru_info["passthrough_logit_margin"] = margin
        mingru_info["passthrough_logit_margin_floor"] = margin_floor
        mingru_info["passthrough_applied"] = applied
        mingru_info["passthrough_base_proposal"] = asdict(proposal)
        if not applied:
            self.last_mingru_terminal = mingru_info
            env["mingru_passthrough"] = dict(mingru_info)
            return proposal
        terminal_force = float(np.clip(prediction.force, -self.config.force_mag, self.config.force_mag))
        passthrough = ForceProposal(
            source_node="mingru_terminal",
            force=terminal_force,
            confidence=max(confidence, proposal.confidence),
            urgency=proposal.urgency,
            reason=f"mingru_terminal_passthrough; base={proposal.source_node}",
            score=max(proposal.score, max(0.01, confidence) * (1.0 + proposal.urgency)),
            raw_confidence=confidence,
            raw_urgency=proposal.raw_urgency,
            select_edge_weight=proposal.select_edge_weight,
            proposal_edge_weight=proposal.proposal_edge_weight,
            bandit_score=proposal.bandit_score,
            selection_multiplier=proposal.selection_multiplier,
            selected=proposal.selected,
            suppressed=False,
            selection_mode=proposal.selection_mode,
        )
        mingru_info["passthrough_force"] = terminal_force
        mingru_info["passthrough_base_force"] = proposal.force
        mingru_info["passthrough_proposal"] = asdict(passthrough)
        self.last_mingru_terminal = mingru_info
        env["mingru_passthrough"] = dict(mingru_info)
        return passthrough

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
        rescue = self.config.rescue
        if abs(features.x) > 1.5:
            if not (self._rescue_enabled() and rescue.rail_vs_pole_priority_gate) or self._rail_imminent(features):
                return "avoid_rail"
        if (
            self._rescue_enabled()
            and rescue.late_episode_conservative_mode
            and self.episode_step >= rescue.late_start_step
            and abs(features.x) < rescue.late_rail_guard
            and self.config.n_poles > 1
        ):
            return "stabilize_chain"
        if goal_vector.get("max_velocity_pressure", 0.0) > 0.65:
            return "damp_energy"
        if self.config.n_poles > 1 and features.worst_pole_index > 0:
            return "stabilize_chain"
        return "recover_worst_pole"
