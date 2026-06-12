from __future__ import annotations

from typing import Any

import numpy as np

from recon_cartpole.control.goal_vector import compute_cartpole_goal_vector
from recon_cartpole.control.policy_observation import adjacent_subchain_features
from recon_cartpole.control.scripts import ProposalGains, REGIMES, propose_force_for_regime
from recon_cartpole.control.sensors import features_from_state

BASIC_RESIDUAL_FEATURE_NAMES = [
    "base_force_norm",
    "risk_gate",
    "previous_force_norm",
]

PROPOSAL_DIAGNOSTIC_FEATURE_NAMES = BASIC_RESIDUAL_FEATURE_NAMES + [
    "chain_force_norm",
    "recover_worst_force_norm",
    "recover_base_force_norm",
    "avoid_rail_force_norm",
    "center_cart_force_norm",
    "damp_energy_force_norm",
    "base_minus_chain_force_norm",
    "base_minus_recover_worst_force_norm",
    "cart_centering_need",
    "max_angle_pressure",
    "max_velocity_pressure",
    "goal_urgency",
    "worst_pole_index_norm",
    "episode_fraction",
]


SUBCHAIN_DIAGNOSTIC_FEATURE_NAMES = PROPOSAL_DIAGNOSTIC_FEATURE_NAMES + [
    "pair01_delta_angle",
    "pair01_delta_velocity",
    "pair01_mean_angle",
    "pair01_mean_velocity",
    "pair12_delta_angle",
    "pair12_delta_velocity",
    "pair12_mean_angle",
    "pair12_mean_velocity",
    "pair23_delta_angle",
    "pair23_delta_velocity",
    "pair23_mean_angle",
    "pair23_mean_velocity",
]


def residual_aux_feature_names(mode: str = "basic") -> list[str]:
    if str(mode) == "proposal_diagnostics":
        return list(PROPOSAL_DIAGNOSTIC_FEATURE_NAMES)
    if str(mode) == "subchain_diagnostics":
        return list(SUBCHAIN_DIAGNOSTIC_FEATURE_NAMES)
    return list(BASIC_RESIDUAL_FEATURE_NAMES)


def residual_aux_feature_size(mode: str = "basic") -> int:
    return len(residual_aux_feature_names(mode))


def residual_risk_gate(raw_state: Any, n_poles: int, horizon: int = 500, episode_step: int = 0) -> float:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    n = int(n_poles)
    if raw.size < 2 + 2 * n:
        return 0.0
    x = abs(float(raw[0])) / 2.4
    theta = float(np.max(np.abs(raw[2 : 2 + n]))) / 0.20943951023931953
    theta_dot = float(np.max(np.abs(raw[2 + n : 2 + 2 * n]))) / 5.0
    late = float(episode_step) / max(1, int(horizon))
    return float(np.clip(max(x, theta, 0.5 * theta_dot, late if late > 0.75 else 0.0), 0.0, 1.0))


def residual_aux_features(
    raw_state: Any,
    *,
    n_poles: int,
    force_mag: float,
    base_force: float,
    previous_force: float = 0.0,
    horizon: int = 500,
    episode_step: int = 0,
    mode: str = "basic",
    proposal_gains: ProposalGains | None = None,
) -> np.ndarray:
    force_scale = max(float(force_mag), 1e-9)
    gate = residual_risk_gate(raw_state, n_poles, horizon, episode_step)
    values = [float(base_force) / force_scale, gate, float(previous_force) / force_scale]
    feature_mode = str(mode)
    if feature_mode not in ("proposal_diagnostics", "subchain_diagnostics"):
        return np.asarray(values, dtype=np.float32)

    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if raw.size < 2 + 2 * int(n_poles):
        target = SUBCHAIN_DIAGNOSTIC_FEATURE_NAMES if feature_mode == "subchain_diagnostics" else PROPOSAL_DIAGNOSTIC_FEATURE_NAMES
        values.extend([0.0] * (len(target) - len(values)))
        return np.asarray(values, dtype=np.float32)

    features = features_from_state(raw, raw, int(n_poles))
    gains = proposal_gains or ProposalGains()
    proposals = {
        regime: propose_force_for_regime(regime, features, float(force_mag), gains)
        for regime in REGIMES
    }
    goal = compute_cartpole_goal_vector(features, n_poles=int(n_poles))
    chain = float(proposals["stabilize_chain"].force)
    recover = float(proposals["recover_worst_pole"].force)
    worst_norm = 0.0
    if int(n_poles) > 1:
        worst_norm = float(features.worst_pole_index) / float(int(n_poles) - 1)
    values.extend(
        [
            chain / force_scale,
            recover / force_scale,
            float(proposals["recover_base_pole"].force) / force_scale,
            float(proposals["avoid_rail"].force) / force_scale,
            float(proposals["center_cart"].force) / force_scale,
            float(proposals["damp_energy"].force) / force_scale,
            (float(base_force) - chain) / force_scale,
            (float(base_force) - recover) / force_scale,
            float(goal.get("cart_centering_need", 0.0)),
            float(goal.get("max_angle_pressure", 0.0)),
            float(goal.get("max_velocity_pressure", 0.0)),
            float(goal.get("urgency", 0.0)),
            worst_norm,
            float(episode_step) / max(1.0, float(horizon)),
        ]
    )
    if feature_mode == "subchain_diagnostics":
        n = int(n_poles)
        theta = raw[2 : 2 + n] / 0.20943951023931953
        theta_dot = raw[2 + n : 2 + 2 * n] / 5.0
        values.extend(adjacent_subchain_features(theta, theta_dot, 4))
    return np.asarray(values, dtype=np.float32)
