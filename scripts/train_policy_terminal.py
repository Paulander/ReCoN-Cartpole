from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout


class UprightShapingWrapper(gym.Wrapper):
    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = np.asarray(info.get("raw_state", []), dtype=float)
        if raw.size >= 2 + 2 * self.env.config.n_poles:
            n = self.env.config.n_poles
            x = float(raw[0]) / max(self.env.config.x_threshold, 1e-9)
            theta = raw[2 : 2 + n] / max(self.env.config.theta_threshold_radians, 1e-9)
            theta_dot = raw[2 + n : 2 + 2 * n] / 5.0
            shaped = 1.0 - 0.35 * float(np.mean(theta * theta)) - 0.05 * x * x - 0.02 * float(np.mean(theta_dot * theta_dot))
            reward = max(-1.0, shaped)
        if terminated:
            reward = -1.0
        return obs, float(reward), terminated, truncated, info


def make_env(args: argparse.Namespace, reward_mode: str = "survival"):
    env = CartPoleNEnv(
        CartPoleNConfig(
            n_poles=args.n_poles,
            horizon=args.horizon,
            dt=args.dt,
            dynamics_mode=args.dynamics_mode,
            action_mode=args.action_mode,
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            initial_angle_range=args.initial_angle_range,
            force_noise=args.force_noise,
            link_coupling=args.link_coupling,
        )
    )
    if reward_mode == "upright_shaping":
        return UprightShapingWrapper(env)
    return env


def evaluate_model(model: Any, args: argparse.Namespace, seeds: list[int]) -> dict[str, Any]:
    steps: list[float] = []
    returns: list[float] = []
    for seed in seeds:
        env = make_env(args, reward_mode="survival")
        obs, _info = env.reset(seed=seed)
        total = 0.0
        for step in range(args.horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            if terminated or truncated:
                steps.append(float(step + 1))
                returns.append(total)
                break
        else:
            steps.append(float(args.horizon))
            returns.append(total)
    summary = summarize_steps(steps, args.horizon)
    summary.update({"returns_mean": float(np.mean(returns)) if returns else 0.0, "episodes": len(seeds)})
    return summary


def evaluate_recon_terminal(model_path: Path, args: argparse.Namespace, seeds: list[int], trace_seed: int | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_policy_terminal",
            action_mode=args.action_mode,
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=str(model_path),
        )
    )
    steps: list[float] = []
    returns: list[float] = []
    for seed in seeds:
        result = rollout(make_env(args, reward_mode="survival"), controller, seed=seed, horizon=args.horizon, trace=False)
        steps.append(float(result["steps"]))
        returns.append(float(result["return"]))
    summary = summarize_steps(steps, args.horizon)
    summary.update({"returns_mean": float(np.mean(returns)) if returns else 0.0, "episodes": len(seeds)})
    if trace_seed is not None and out_dir is not None:
        trace_result = rollout(make_env(args, reward_mode="survival"), controller, seed=trace_seed, horizon=args.horizon, trace=True)
        (out_dir / "recon_policy_terminal_trace.json").write_text(json.dumps({"steps": trace_result["trace"]}, indent=2), encoding="utf-8")
    return summary


def train_policy_terminal(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install RL extras with `uv sync --extra rl` to train policy terminals") from exc

    if args.model_path:
        model_path = Path(args.model_path)
        model = PPO.load(str(model_path), device=args.device)
        train_timesteps = 0
        status = "evaluated"
    else:
        train_env = make_vec_env(lambda: make_env(args, reward_mode=args.reward_mode), n_envs=args.n_envs, seed=args.train_seed)
        if args.resume_model_path:
            model = PPO.load(str(args.resume_model_path), env=train_env, device=args.device)
            model.set_random_seed(args.train_seed)
            status = "resumed"
        else:
            model = PPO(args.policy, train_env, seed=args.train_seed, verbose=args.verbose, device=args.device)
            status = "completed"
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not bool(args.resume_model_path))
        model_path = out / "ppo_policy_terminal.zip"
        model.save(str(model_path))
        train_timesteps = args.timesteps

    seeds = [args.eval_seed_start + i for i in range(args.eval_episodes)]
    ppo_eval = evaluate_model(model, args, seeds)
    recon_eval = evaluate_recon_terminal(model_path, args, seeds, trace_seed=args.eval_seed_start + 999_999, out_dir=out)
    report = {
        "status": status,
        "model_path": str(model_path),
        "train_timesteps": train_timesteps,
        "train_seed": args.train_seed,
        "eval_seeds": seeds,
        "env": {
            "n_poles": args.n_poles,
            "horizon": args.horizon,
            "dt": args.dt,
            "dynamics_mode": args.dynamics_mode,
            "action_mode": args.action_mode,
            "discrete_action_bins": args.discrete_action_bins,
            "force_mag": args.force_mag,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "link_coupling": args.link_coupling,
        },
        "reward_mode": args.reward_mode,
        "selection_mode": args.selection_mode,
        "pure_ppo_eval": ppo_eval,
        "recon_policy_terminal_eval": recon_eval,
        "mechanisms": {
            "ppo_policy_gradient": True,
            "policy_terminal": True,
            "edge_plasticity": False,
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report_md(report, out / "report.md")
    return report


def write_report_md(report: dict[str, Any], path: Path) -> None:
    ppo = report["pure_ppo_eval"]
    recon = report["recon_policy_terminal_eval"]
    lines = [
        "# Policy Terminal Training Report",
        "",
        f"Status: `{report['status']}`",
        f"Reward mode: `{report['reward_mode']}`",
        f"Model: `{report['model_path']}`",
        f"Train timesteps: `{report['train_timesteps']}`",
        f"Wall-clock seconds: `{report['wall_clock_seconds']:.2f}`",
        "",
        "| evaluator | mean | p10 | success | max | episodes |",
        "|---|---:|---:|---:|---:|---:|",
        f"| pure_ppo | {ppo['mean_survival']:.1f} | {ppo['p10_survival']:.1f} | {ppo['success_rate']:.2f} | {ppo['max_survival']:.1f} | {ppo['episodes']} |",
        f"| recon_policy_terminal | {recon['mean_survival']:.1f} | {recon['p10_survival']:.1f} | {recon['success_rate']:.2f} | {recon['max_survival']:.1f} | {recon['episodes']} |",
        "",
        "## Claim Discipline",
        "",
        "This report separates a learned PPO policy from ReCoN's graph scaffold. It is not pure symbolic ReCoN, and it is not a solved claim unless the held-out solve thresholds are met.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="parallel")
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--model-path", default="", help="Evaluate an existing PPO policy zip instead of training a new one.")
    parser.add_argument("--resume-model-path", default="", help="Continue training an existing PPO policy zip, then save a new terminal policy.")
    parser.add_argument("--train-seed", type=int, default=510_000)
    parser.add_argument("--eval-seed-start", type=int, default=620_000)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", default="MlpPolicy")
    parser.add_argument("--reward-mode", choices=["survival", "upright_shaping"], default="survival")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--out", default="reports/policy_terminal_train")
    args = parser.parse_args()
    result = train_policy_terminal(args)
    print(json.dumps({"out": args.out, "model_path": result["model_path"], "wall_clock_seconds": result["wall_clock_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
