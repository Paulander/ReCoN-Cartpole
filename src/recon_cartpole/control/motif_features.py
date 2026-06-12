from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.policy_observation import adjacent_subchain_features


def load_motif_model(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    value = str(path).strip()
    if not value:
        return None
    return json.loads(Path(value).read_text(encoding="utf-8"))


def motif_feature_vector(
    raw_state: Any,
    n_poles: int,
    model: dict[str, Any],
    force_mag: float = 10.0,
    base_force: float = 0.0,
    x_threshold: float = 2.4,
    theta_threshold: float = 12.0 * 2.0 * np.pi / 360.0,
    cart_velocity_scale: float = 5.0,
    pole_velocity_scale: float = 5.0,
) -> np.ndarray:
    raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    n = int(n_poles)
    needed = 2 + 2 * n
    if raw.size < needed:
        raise ValueError("raw_state is too short for motif features")
    theta = raw[2 : 2 + n] / max(float(theta_threshold), 1e-9)
    theta_dot = raw[2 + n : 2 + 2 * n] / max(float(pole_velocity_scale), 1e-9)
    values = [
        float(raw[0]) / max(float(x_threshold), 1e-9),
        float(raw[1]) / max(float(cart_velocity_scale), 1e-9),
    ]
    values.extend(adjacent_subchain_features(theta, theta_dot, max(4, n)))
    target_dim = len(model.get("scale", values))
    if target_dim > len(values):
        diagnostics = [
            float(base_force) / max(float(force_mag), 1e-9),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        values.extend(diagnostics[: target_dim - len(values)])
    if len(values) < target_dim:
        values.extend([0.0] * (target_dim - len(values)))
    return np.asarray(values[:target_dim], dtype=np.float32)


def motif_score(model: dict[str, Any], vector: Any) -> float:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    pos = np.asarray(model["positive_mean"], dtype=np.float32)
    neg = np.asarray(model["negative_mean"], dtype=np.float32)
    scale = np.maximum(np.asarray(model["scale"], dtype=np.float32), 1e-6)
    z = vec / scale
    p = pos / scale
    n = neg / scale
    d_pos = float(np.mean((z - p) ** 2))
    d_neg = float(np.mean((z - n) ** 2))
    return d_neg - d_pos


def motif_score_from_state(
    raw_state: Any,
    n_poles: int,
    model: dict[str, Any] | None,
    force_mag: float = 10.0,
    base_force: float = 0.0,
) -> float:
    if model is None:
        return 0.0
    return motif_score(
        model,
        motif_feature_vector(
            raw_state,
            n_poles=n_poles,
            model=model,
            force_mag=force_mag,
            base_force=base_force,
        ),
    )
