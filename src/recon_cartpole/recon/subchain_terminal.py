from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.sensors import StateFeatures


@dataclass
class SubchainTerminalConfig:
    enabled: bool = False
    checkpoint_path: str = ""
    blend: float = 0.35
    hidden_size: int = 32
    theta_scale: float = 0.21
    velocity_scale: float = 5.0
    x_scale: float = 2.4
    force_mag: float = 10.0
    min_pair_pressure: float = 0.02
    min_confidence: float = 0.0
    max_force_fraction: float = 1.0
    include_pair_position: bool = True
    confidence_boost: float = 0.08
    urgency_boost: float = 0.15
    regimes: tuple[str, ...] = ("stabilize_chain",)


@dataclass
class SubchainVote:
    pair: int
    force: float
    confidence: float
    pressure: float
    weight: float
    features: list[float] = field(default_factory=list)


class SharedSubchainTerminal:
    """Shared learned pair terminal applied to each adjacent pole pair."""

    def __init__(self, n_poles: int, force_mag: float, config: SubchainTerminalConfig | None = None):
        self.n_poles = int(n_poles)
        self.force_mag = float(force_mag)
        self.config = config or SubchainTerminalConfig()
        self.config.force_mag = self.force_mag
        self.model: Any | None = None
        self.loaded_checkpoint = ""
        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)

    @property
    def input_size(self) -> int:
        return 7 if self.config.include_pair_position else 6

    def _torch(self):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("learned subchain terminals require torch") from exc
        return torch

    def build_model(self, hidden_size: int | None = None):
        self._torch()
        import torch.nn as nn

        hidden = max(1, int(hidden_size or self.config.hidden_size))
        return nn.Sequential(
            nn.Linear(self.input_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def load_checkpoint(self, path: str) -> None:
        torch = self._torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = dict(payload.get("meta", {})) if isinstance(payload, dict) else {}
        if meta:
            self.config.hidden_size = int(meta.get("hidden_size", self.config.hidden_size))
            self.config.include_pair_position = bool(meta.get("include_pair_position", self.config.include_pair_position))
            self.config.theta_scale = float(meta.get("theta_scale", self.config.theta_scale))
            self.config.velocity_scale = float(meta.get("velocity_scale", self.config.velocity_scale))
            self.config.x_scale = float(meta.get("x_scale", self.config.x_scale))
        self.model = self.build_model(self.config.hidden_size)
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        self.model.load_state_dict(state)
        self.model.eval()
        self.loaded_checkpoint = str(path)

    def pair_feature_vector(self, features: StateFeatures, pair_index: int) -> tuple[np.ndarray, float]:
        left = features.poles[pair_index]
        right = features.poles[pair_index + 1]
        theta_scale = max(float(self.config.theta_scale), 1e-9)
        velocity_scale = max(float(self.config.velocity_scale), 1e-9)
        values = [
            float(features.x) / max(float(self.config.x_scale), 1e-9),
            float(features.x_dot) / velocity_scale,
            float(right.theta - left.theta) / theta_scale,
            float(right.theta_dot - left.theta_dot) / velocity_scale,
            float(0.5 * (left.theta + right.theta)) / theta_scale,
            float(0.5 * (left.theta_dot + right.theta_dot)) / velocity_scale,
        ]
        if self.config.include_pair_position:
            denom = max(1, self.n_poles - 2)
            values.append(float(pair_index) / float(denom))
        pressure = max(abs(v) for v in values[2:6])
        return np.asarray(values, dtype=np.float32), float(pressure)

    def votes(self, features: StateFeatures) -> list[SubchainVote]:
        if self.model is None or len(features.poles) < 2:
            return []
        torch = self._torch()
        vectors: list[np.ndarray] = []
        pressures: list[float] = []
        for pair in range(len(features.poles) - 1):
            vec, pressure = self.pair_feature_vector(features, pair)
            vectors.append(vec)
            pressures.append(pressure)
        x = torch.as_tensor(np.stack(vectors), dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x)
        raw = out.detach().cpu().numpy()
        force_limit = self.force_mag * max(0.0, min(1.0, float(self.config.max_force_fraction)))
        result: list[SubchainVote] = []
        for pair, row in enumerate(raw):
            force = float(np.tanh(row[0]) * force_limit)
            confidence = float(1.0 / (1.0 + np.exp(-row[1])))
            pressure = float(pressures[pair])
            active = pressure >= float(self.config.min_pair_pressure) and confidence >= float(self.config.min_confidence)
            weight = float(confidence * max(0.0, pressure)) if active else 0.0
            result.append(
                SubchainVote(
                    pair=pair,
                    force=force,
                    confidence=confidence,
                    pressure=pressure,
                    weight=weight,
                    features=[float(v) for v in vectors[pair].tolist()],
                )
            )
        return result


def save_subchain_terminal_checkpoint(
    path: str | Path,
    model: Any,
    config: SubchainTerminalConfig,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "hidden_size": int(config.hidden_size),
        "include_pair_position": bool(config.include_pair_position),
        "theta_scale": float(config.theta_scale),
        "velocity_scale": float(config.velocity_scale),
        "x_scale": float(config.x_scale),
    }
    if extra_meta:
        meta.update(extra_meta)
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, out)
