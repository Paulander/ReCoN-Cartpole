from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.control.policy_observation import policy_observation_from_state, policy_observation_size
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.training.ablations import summarize_steps
from train_policy_terminal import load_observation_normalizer, parse_seed_list


class ResidualCorrectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, args: argparse.Namespace, *, hard_seeds: list[int] | None = None):
        super().__init__()
        try:
            from stable_baselines3 import PPO
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install RL extras with `uv sync --extra rl`") from exc
        self.args = args
        self.env = CartPoleNEnv(
            CartPoleNConfig(
                n_poles=args.n_poles,
                horizon=args.horizon,
                dt=args.dt,
                dynamics_mode=args.dynamics_mode,
                action_mode=args.env_action_mode,
                discrete_action_bins=args.discrete_action_bins,
                force_mag=args.force_mag,
                initial_angle_range=args.initial_angle_range,
                force_noise=args.force_noise,
                link_coupling=args.link_coupling,
            )
        )
        self.base_model = PPO.load(str(args.base_model_path), device=args.device)
        self.base_normalizer = (
            load_observation_normalizer(args.base_normalizer_path)
            if args.base_normalizer_path
            else None
        )
        self.hard_seeds = list(hard_seeds or [])
        self.seed_index = 0
        self.previous_force = 0.0
        self.step_count = 0
        self.last_raw: np.ndarray | None = None
        self.last_env_obs: Any | None = None
        self.action_space = gym.spaces.Discrete(max(2, int(args.residual_action_bins)))
        base_size = policy_observation_size(args.n_poles, args.base_observation_mode)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(base_size + 3,), dtype=np.float32
        )

    def _base_policy_obs(self, env_obs: Any, raw: Any) -> np.ndarray:
        obs = policy_observation_from_state(
            env_obs,
            raw,
            self.args.n_poles,
            self.args.base_observation_mode,
            previous_force=self.previous_force,
            force_mag=self.args.force_mag,
        )
        if self.base_normalizer is not None:
            mean = self.base_normalizer["mean"]
            var = self.base_normalizer["var"]
            obs = np.clip(
                (obs - mean) / np.sqrt(var + float(self.base_normalizer["epsilon"])),
                -float(self.base_normalizer["clip_obs"]),
                float(self.base_normalizer["clip_obs"]),
            ).astype(np.float32)
        return obs

    def _base_force(self, env_obs: Any, raw: Any) -> float:
        action, _state = self.base_model.predict(self._base_policy_obs(env_obs, raw), deterministic=True)
        if self.args.env_action_mode == "continuous":
            return float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -self.args.force_mag, self.args.force_mag))
        bins = max(2, int(self.args.discrete_action_bins))
        idx = int(np.clip(int(np.asarray(action).reshape(-1)[0]), 0, bins - 1))
        if bins == 2:
            return float(self.args.force_mag if idx == 1 else -self.args.force_mag)
        return float(np.linspace(-self.args.force_mag, self.args.force_mag, bins)[idx])

    def _risk_gate(self, raw: Any) -> float:
        raw_arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        n = int(self.args.n_poles)
        if raw_arr.size < 2 + 2 * n:
            return 0.0
        x = abs(float(raw_arr[0])) / 2.4
        theta = np.max(np.abs(raw_arr[2 : 2 + n])) / 0.20943951023931953
        theta_dot = np.max(np.abs(raw_arr[2 + n : 2 + 2 * n])) / 5.0
        late = self.step_count / max(1, int(self.args.horizon))
        return float(np.clip(max(x, theta, 0.5 * theta_dot, late if late > 0.75 else 0.0), 0.0, 1.0))

    def _residual_obs(self, env_obs: Any, raw: Any) -> np.ndarray:
        base_obs = self._base_policy_obs(env_obs, raw)
        base_force = self._base_force(env_obs, raw)
        gate = self._risk_gate(raw)
        return np.concatenate(
            [base_obs, np.asarray([base_force / self.args.force_mag, gate, self.previous_force / self.args.force_mag], dtype=np.float32)]
        ).astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.previous_force = 0.0
        self.step_count = 0
        if self.hard_seeds and self.np_random.random() < float(self.args.hard_seed_probability):
            seed = self.hard_seeds[self.seed_index % len(self.hard_seeds)]
            self.seed_index += 1
        env_obs, info = self.env.reset(seed=seed, options=options)
        self.last_raw = np.asarray(info.get("raw_state"), dtype=np.float32)
        self.last_env_obs = env_obs
        return self._residual_obs(env_obs, self.last_raw), info

    def step(self, action: Any):
        raw = self.last_raw
        assert raw is not None
        base_force = self._base_force(self.last_env_obs if self.last_env_obs is not None else raw, raw)
        bins = np.linspace(-self.args.max_residual_force, self.args.max_residual_force, self.action_space.n)
        delta = float(bins[int(np.clip(int(np.asarray(action).reshape(-1)[0]), 0, self.action_space.n - 1))])
        gate = self._risk_gate(raw)
        force = float(np.clip(base_force + gate * delta, -self.args.force_mag, self.args.force_mag))
        env_action = action_from_force(force, self.args.env_action_mode, self.args.force_mag, self.args.discrete_action_bins)
        env_obs, reward, terminated, truncated, info = self.env.step(env_action)
        self.step_count += 1
        self.previous_force = force
        self.last_raw = np.asarray(info.get("raw_state"), dtype=np.float32)
        self.last_env_obs = env_obs
        change_penalty = float(self.args.low_risk_change_penalty) * abs(delta / max(self.args.max_residual_force, 1e-9)) * (1.0 - gate)
        late_bonus = float(self.args.late_survival_bonus) if self.step_count >= int(self.args.horizon * self.args.late_survival_start_fraction) else 0.0
        shaped_reward = float(reward) + late_bonus - change_penalty
        info = {**info, "base_force": base_force, "residual_delta": delta, "risk_gate": gate, "final_force": force}
        return self._residual_obs(env_obs, self.last_raw), shaped_reward, terminated, truncated, info


def make_env(args: argparse.Namespace, hard_seeds: list[int] | None = None):
    return ResidualCorrectionEnv(args, hard_seeds=hard_seeds)


def evaluate_base(args: argparse.Namespace, seeds: list[int]) -> dict[str, Any]:
    env = ResidualCorrectionEnv(args)
    steps: list[float] = []
    for seed in seeds:
        obs, _info = env.reset(seed=seed)
        total_steps = 0
        for step in range(args.horizon):
            # middle residual bin is zero delta when bins are odd
            action = env.action_space.n // 2
            obs, _reward, terminated, truncated, _info = env.step(action)
            total_steps = step + 1
            if terminated or truncated:
                break
        steps.append(float(total_steps))
    summary = summarize_steps(steps, args.horizon)
    summary["episodes"] = len(seeds)
    return summary


def evaluate_residual(model: Any, args: argparse.Namespace, seeds: list[int]) -> dict[str, Any]:
    steps: list[float] = []
    deltas: list[float] = []
    for seed in seeds:
        env = ResidualCorrectionEnv(args)
        obs, _info = env.reset(seed=seed)
        total_steps = 0
        for step in range(args.horizon):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            deltas.append(abs(float(info.get("residual_delta", 0.0))))
            total_steps = step + 1
            if terminated or truncated:
                break
        steps.append(float(total_steps))
    summary = summarize_steps(steps, args.horizon)
    summary.update({"episodes": len(seeds), "mean_abs_residual_delta": float(np.mean(deltas)) if deltas else 0.0})
    return summary


def write_markdown(report: dict[str, Any], path: Path) -> None:
    base = report["base_eval"]
    residual = report["residual_eval"]
    lines = [
        "# Residual Policy Terminal Training",
        "",
        f"Status: `{report['status']}`",
        f"Base model: `{report['base_model_path']}`",
        f"Residual model: `{report['residual_model_path']}`",
        "",
        "| evaluator | mean | p10 | success | max | episodes |",
        "|---|---:|---:|---:|---:|---:|",
        f"| frozen_base | {base['mean_survival']:.1f} | {base['p10_survival']:.1f} | {base['success_rate']:.3f} | {base['max_survival']:.1f} | {base['episodes']} |",
        f"| residual_specialist | {residual['mean_survival']:.1f} | {residual['p10_survival']:.1f} | {residual['success_rate']:.3f} | {residual['max_survival']:.1f} | {residual['episodes']} |",
        "",
        "## Mechanisms",
        "",
        "The base PPO terminal is frozen. The residual learner sees base force, previous force, and a risk gate; low-risk changes are penalized so the specialist focuses on late/tail failures rather than rewriting successful behavior.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_residual(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install RL extras with `uv sync --extra rl`") from exc
    hard_seeds = parse_seed_list(args.hard_train_seeds)
    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else None
    train_env = make_vec_env(
        lambda: make_env(args, hard_seeds=hard_seeds),
        n_envs=args.n_envs,
        seed=args.train_seed,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs={"start_method": "fork"} if vec_env_cls is SubprocVecEnv else None,
    )
    policy_kwargs = {"net_arch": [int(item) for item in args.net_arch.split(",") if item.strip()]}
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=args.train_seed,
        verbose=args.verbose,
        device=args.device,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=args.timesteps)
    model_path = out / "residual_policy_terminal.zip"
    model.save(str(model_path))
    seeds = [args.eval_seed_start + idx for idx in range(args.eval_episodes)]
    report = {
        "status": "completed",
        "base_model_path": args.base_model_path,
        "base_normalizer_path": args.base_normalizer_path,
        "residual_model_path": str(model_path),
        "timesteps": args.timesteps,
        "eval_seeds": seeds,
        "base_eval": evaluate_base(args, seeds),
        "residual_eval": evaluate_residual(model, args, seeds),
        "mechanisms": {
            "frozen_base_policy_terminal": True,
            "learned_residual_policy": True,
            "risk_gate": True,
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out / "report.md")
    train_env.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--base-normalizer-path", default="")
    parser.add_argument("--base-observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force"], default="normalized_raw")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--env-action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--residual-action-bins", type=int, default=5)
    parser.add_argument("--max-residual-force", type=float, default=4.0)
    parser.add_argument("--low-risk-change-penalty", type=float, default=0.05)
    parser.add_argument("--late-survival-bonus", type=float, default=0.02)
    parser.add_argument("--late-survival-start-fraction", type=float, default=0.80)
    parser.add_argument("--hard-train-seeds", default="")
    parser.add_argument("--hard-seed-probability", type=float, default=0.55)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--train-seed", type=int, default=2_400_000)
    parser.add_argument("--eval-seed-start", type=int, default=1_040_000)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--net-arch", default="64,64")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.03)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--out", default="reports/residual_policy_terminal")
    args = parser.parse_args()
    report = train_residual(args)
    print(json.dumps({"out": args.out, "status": report["status"], "residual_model_path": report["residual_model_path"]}, indent=2))


if __name__ == "__main__":
    main()
