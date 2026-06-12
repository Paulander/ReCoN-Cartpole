from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.motif_features import load_motif_model, motif_score_from_state
from recon_cartpole.control.policy_observation import policy_observation_from_state


@dataclass
class MinGRUTerminalConfig:
    enabled: bool = False
    hidden_size: int = 64
    sequence_length: int = 8
    observation_mode: str = "normalized_raw"
    include_prev_force: bool = True
    include_context: bool = True
    include_motif_score: bool = False
    motif_model_path: str = ""
    motif_score_scale: float = 10.0
    blend: float = 1.0
    scope: str = "stabilize_chain"
    confidence_floor: float = 0.05
    passthrough_enabled: bool = False
    passthrough_confidence_floor: float = 0.05
    passthrough_logit_margin_floor: float = 0.0
    checkpoint_path: str = ""


@dataclass
class MinGRUPrediction:
    force: float | None
    confidence: float
    value: float
    failure_probability: float
    hidden_norm: float
    sequence_length: int
    logits: list[float] = field(default_factory=list)
    valid: bool = True
    reason: str = ""


class MinGRUTerminal:
    """Small recurrent terminal that emits a force proposal for ReCoN arbitration."""

    def __init__(
        self,
        n_poles: int,
        force_mag: float,
        discrete_action_bins: int,
        config: MinGRUTerminalConfig | None = None,
    ):
        self.config = config or MinGRUTerminalConfig()
        self.n_poles = int(n_poles)
        self.force_mag = float(force_mag)
        self.discrete_action_bins = max(2, int(discrete_action_bins))
        self.obs_history: list[np.ndarray] = []
        self.prev_force = 0.0
        self.model: Any | None = None
        self.hidden: Any | None = None
        self.input_size: int | None = None
        self.loaded_checkpoint = ""
        self.motif_model: dict[str, Any] | None = load_motif_model(self.config.motif_model_path)
        self._build_model()
        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)

    def _torch(self):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("recon_mingru_terminal requires torch") from exc
        return torch

    def _build_model(self) -> None:
        torch = self._torch()
        self.input_size = self._input_size()
        self.model = _MinGRUNetwork(
            input_size=self.input_size,
            hidden_size=max(1, int(self.config.hidden_size)),
            action_bins=self.discrete_action_bins,
        )
        self.model.eval()
        self.hidden = torch.zeros(1, max(1, int(self.config.hidden_size)), dtype=torch.float32)

    def _base_observation(self, observation: Any, raw_state: Any | None) -> np.ndarray:
        return policy_observation_from_state(
            observation,
            raw_state,
            self.n_poles,
            self.config.observation_mode,
            previous_force=self.prev_force,
            force_mag=self.force_mag,
        ).astype(np.float32, copy=False)

    def _context_vector(self, context: dict[str, Any] | None) -> list[float]:
        if not self.config.include_context:
            return []
        if not context:
            return [0.0, 0.0, 0.0]
        goal = context.get("goal_vector", {}) or {}
        selected = context.get("selected_regime")
        selected_flag = 1.0 if selected == "stabilize_chain" else 0.0
        return [
            float(goal.get("risk", 0.0)),
            float(goal.get("max_velocity_pressure", 0.0)),
            selected_flag,
        ]

    def _input_size(self) -> int:
        base = policy_observation_from_state(
            np.zeros(2 + 3 * self.n_poles, dtype=np.float32),
            np.zeros(2 + 2 * self.n_poles, dtype=np.float32),
            self.n_poles,
            self.config.observation_mode,
        ).size
        extras = 1 if self.config.include_prev_force and "prev_force" not in self.config.observation_mode else 0
        extras += 3 if self.config.include_context else 0
        extras += 1 if self.config.include_motif_score else 0
        return int(base + extras)

    def reset(self) -> None:
        torch = self._torch()
        self.obs_history = []
        self.prev_force = 0.0
        self.hidden = torch.zeros(1, max(1, int(self.config.hidden_size)), dtype=torch.float32)

    def load_checkpoint(self, path: str) -> None:
        torch = self._torch()
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        if isinstance(checkpoint_config, dict):
            runtime_control = {
                "enabled": self.config.enabled,
                "blend": self.config.blend,
                "scope": self.config.scope,
                "confidence_floor": self.config.confidence_floor,
                "passthrough_enabled": self.config.passthrough_enabled,
                "passthrough_confidence_floor": self.config.passthrough_confidence_floor,
                "passthrough_logit_margin_floor": self.config.passthrough_logit_margin_floor,
                "checkpoint_path": self.config.checkpoint_path,
            }
            for key, value in checkpoint_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            for key, value in runtime_control.items():
                setattr(self.config, key, value)
            self.motif_model = load_motif_model(self.config.motif_model_path)
            self._build_model()
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        assert self.model is not None
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.loaded_checkpoint = str(path)

    def save_checkpoint(self, path: str) -> None:
        torch = self._torch()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.__dict__,
                "n_poles": self.n_poles,
                "force_mag": self.force_mag,
                "discrete_action_bins": self.discrete_action_bins,
            },
            out,
        )

    def observation_vector(
        self, observation: Any, raw_state: Any | None, context: dict[str, Any] | None = None
    ) -> np.ndarray:
        parts = [self._base_observation(observation, raw_state)]
        if self.config.include_prev_force and "prev_force" not in self.config.observation_mode:
            parts.append(np.asarray([self.prev_force / max(self.force_mag, 1e-9)], dtype=np.float32))
        context_values = self._context_vector(context)
        if context_values:
            parts.append(np.asarray(context_values, dtype=np.float32))
        if self.config.include_motif_score:
            score = motif_score_from_state(
                raw_state,
                self.n_poles,
                self.motif_model,
                force_mag=self.force_mag,
                base_force=self.prev_force,
            )
            parts.append(np.asarray([score / max(float(self.config.motif_score_scale), 1e-9)], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32, copy=False)

    def predict(
        self, observation: Any, raw_state: Any | None, context: dict[str, Any] | None = None
    ) -> MinGRUPrediction:
        if self.model is None:
            return MinGRUPrediction(None, 0.0, 0.0, 0.0, 0.0, 0, valid=False, reason="no_model")
        torch = self._torch()
        vector = self.observation_vector(observation, raw_state, context)
        if not np.all(np.isfinite(vector)):
            return MinGRUPrediction(None, 0.0, 0.0, 1.0, 0.0, 0, valid=False, reason="nonfinite_input")
        self.obs_history.append(vector)
        seq_len = max(1, int(self.config.sequence_length))
        self.obs_history = self.obs_history[-seq_len:]
        pad_count = seq_len - len(self.obs_history)
        frames = [self.obs_history[0]] * pad_count + self.obs_history
        x = torch.as_tensor(np.stack(frames), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value, failure, confidence, self.hidden = self.model(x, self.hidden)
        logits_np = logits.detach().cpu().reshape(-1).numpy()
        if not np.all(np.isfinite(logits_np)):
            return MinGRUPrediction(None, 0.0, 0.0, 1.0, 0.0, len(frames), valid=False, reason="nonfinite_output")
        action_idx = int(np.argmax(logits_np))
        force = float(np.linspace(-self.force_mag, self.force_mag, self.discrete_action_bins)[action_idx])
        confidence_value = float(confidence.detach().cpu().reshape(-1)[0])
        value_value = float(value.detach().cpu().reshape(-1)[0])
        failure_value = float(failure.detach().cpu().reshape(-1)[0])
        hidden_norm = float(torch.linalg.vector_norm(self.hidden.detach()).cpu())
        self.prev_force = force
        return MinGRUPrediction(
            force=force,
            confidence=max(0.0, min(1.0, confidence_value)),
            value=value_value,
            failure_probability=max(0.0, min(1.0, failure_value)),
            hidden_norm=hidden_norm,
            sequence_length=len(frames),
            logits=[float(v) for v in logits_np.tolist()],
        )


class _MinGRUNetwork:
    def __new__(cls, input_size: int, hidden_size: int, action_bins: int):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(input_size, hidden_size)
                self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
                self.candidate = nn.Linear(input_size, hidden_size)
                self.policy_head = nn.Linear(hidden_size, action_bins)
                self.value_head = nn.Linear(hidden_size, 1)
                self.failure_head = nn.Linear(hidden_size, 1)
                self.confidence_head = nn.Linear(hidden_size, 1)

            def forward(self, x, hidden):
                import torch

                h = hidden
                for t in range(x.shape[1]):
                    xt = x[:, t, :]
                    gate = torch.sigmoid(self.update_gate(torch.cat([xt, h], dim=-1)))
                    cand = torch.tanh(self.input_proj(xt) + self.candidate(xt))
                    h = (1.0 - gate) * h + gate * cand
                logits = self.policy_head(h)
                value = self.value_head(h)
                failure = torch.sigmoid(self.failure_head(h))
                confidence = torch.sigmoid(self.confidence_head(h))
                return logits, value, failure, confidence, h.detach()

        return Net()
