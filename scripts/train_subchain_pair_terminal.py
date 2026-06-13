from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.sensors import features_from_state
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.subchain_terminal import (
    SharedSubchainTerminal,
    SubchainTerminalConfig,
    save_subchain_terminal_checkpoint,
)
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout


def make_env(args: argparse.Namespace) -> CartPoleNEnv:
    return CartPoleNEnv(
        CartPoleNConfig(
            n_poles=args.n_poles,
            horizon=args.horizon,
            dt=args.dt,
            dynamics_mode=args.dynamics_mode,
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            initial_angle_range=args.initial_angle_range,
            force_noise=args.force_noise,
            link_coupling=args.link_coupling,
        )
    )


def make_teacher(args: argparse.Namespace) -> ReConCartPoleController:
    return ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode=args.teacher_mode,
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=args.policy_terminal_path,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.policy_terminal_observation_mode,
        )
    )


def collect_dataset(args: argparse.Namespace, terminal: SharedSubchainTerminal) -> dict[str, np.ndarray]:
    xs: list[np.ndarray] = []
    force_targets: list[float] = []
    confidence_targets: list[float] = []
    pair_indices: list[int] = []
    seeds: list[int] = []
    steps: list[int] = []
    teacher = make_teacher(args)
    env = make_env(args)
    for ep in range(int(args.episodes)):
        seed = int(args.seed_start) + ep
        obs, info = env.reset(seed=seed)
        teacher.start_episode()
        for step in range(int(args.horizon)):
            raw = info.get("raw_state")
            action, diagnostics = teacher.act(obs, raw)
            force = float(diagnostics.get("force", 0.0))
            features = features_from_state(obs, raw, args.n_poles)
            for pair in range(max(0, len(features.poles) - 1)):
                vec, pressure = terminal.pair_feature_vector(features, pair)
                xs.append(vec)
                force_targets.append(force / max(float(args.force_mag), 1e-9))
                confidence_targets.append(float(min(1.0, max(0.0, pressure))))
                pair_indices.append(pair)
                seeds.append(seed)
                steps.append(step)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
    if not xs:
        raise ValueError("no subchain pair samples collected")
    return {
        "x": np.stack(xs).astype(np.float32),
        "force_targets": np.asarray(force_targets, dtype=np.float32),
        "confidence_targets": np.asarray(confidence_targets, dtype=np.float32),
        "pair_indices": np.asarray(pair_indices, dtype=np.int64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "steps": np.asarray(steps, dtype=np.int64),
    }


def train_model(args: argparse.Namespace, data: dict[str, np.ndarray], terminal: SharedSubchainTerminal):
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install RL extras with torch to train the subchain pair terminal") from exc
    torch.manual_seed(int(args.train_seed))
    rng = np.random.default_rng(int(args.train_seed))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = terminal.build_model(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    x = torch.as_tensor(data["x"], dtype=torch.float32, device=device)
    force_target = torch.as_tensor(data["force_targets"], dtype=torch.float32, device=device)
    confidence_target = torch.as_tensor(data["confidence_targets"], dtype=torch.float32, device=device)
    indices = np.arange(x.shape[0])
    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        rng.shuffle(indices)
        rows: list[dict[str, float]] = []
        for start in range(0, len(indices), int(args.batch_size)):
            idx = indices[start : start + int(args.batch_size)]
            if idx.size == 0:
                continue
            xb = x[idx]
            out = model(xb)
            pred_force = torch.tanh(out[:, 0])
            pred_conf = torch.sigmoid(out[:, 1])
            force_loss = F.mse_loss(pred_force, force_target[idx])
            confidence_loss = F.binary_cross_entropy(pred_conf, confidence_target[idx])
            loss = force_loss + float(args.confidence_loss_weight) * confidence_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "force_loss": float(force_loss.detach().cpu()),
                    "confidence_loss": float(confidence_loss.detach().cpu()),
                }
            )
        history.append({key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {})
    return model, history


def eval_controller(args: argparse.Namespace, checkpoint_path: str, mode: str) -> dict[str, Any]:
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode=mode,
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            reset_bandit_each_episode=False,
            policy_terminal_path=args.policy_terminal_path,
            policy_terminal_blend=args.policy_terminal_blend,
            policy_terminal_scope=args.policy_terminal_scope,
            policy_terminal_observation_mode=args.policy_terminal_observation_mode,
            learned_subchain_terminal=SubchainTerminalConfig(
                enabled=bool(checkpoint_path),
                checkpoint_path=checkpoint_path,
                blend=args.subchain_blend,
                min_confidence=args.min_confidence,
                min_pair_pressure=args.min_pair_pressure,
                max_force_fraction=args.max_force_fraction,
                confidence_boost=args.confidence_boost,
                urgency_boost=args.urgency_boost,
            ),
        )
    )
    steps: list[float] = []
    for seed in eval_seeds(args):
        result = rollout(make_env(args), controller, seed=int(seed), horizon=args.horizon, trace=False)
        steps.append(float(result["steps"]))
    summary = summarize_steps(steps, args.horizon)
    summary["episodes"] = len(steps)
    return summary


def eval_seeds(args: argparse.Namespace) -> list[int]:
    starts = args.eval_seed_starts or [args.eval_seed_start]
    seeds: list[int] = []
    for start in starts:
        seeds.extend(int(start) + idx for idx in range(int(args.eval_episodes)))
    return seeds


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SubchainTerminalConfig(
        hidden_size=args.hidden_size,
        include_pair_position=not args.no_pair_position,
        theta_scale=args.theta_scale,
        velocity_scale=args.velocity_scale,
        x_scale=args.x_scale,
        force_mag=args.force_mag,
    )
    terminal = SharedSubchainTerminal(args.n_poles, args.force_mag, cfg)
    data = collect_dataset(args, terminal)
    np.savez_compressed(out / "dataset.npz", **data)
    model, history = train_model(args, data, terminal)
    checkpoint = out / "subchain_pair_terminal.pt"
    save_subchain_terminal_checkpoint(checkpoint, model, cfg, {"samples": int(data["x"].shape[0])})
    base_mode = args.teacher_mode if args.teacher_mode == "recon_policy_terminal" else "static_recon"
    base_eval = eval_controller(args, "", base_mode)
    learned_eval = eval_controller(args, str(checkpoint), base_mode)
    report = {
        "status": "completed",
        "checkpoint_path": str(checkpoint),
        "samples": int(data["x"].shape[0]),
        "pairs": int(max(0, args.n_poles - 1)),
        "history": history,
        "base_eval": base_eval,
        "learned_subchain_eval": learned_eval,
        "config": vars(args),
        "mechanisms": {
            "shared_subchain_terminal": True,
            "supervised_teacher_distillation": True,
            "policy_terminal": args.teacher_mode == "recon_policy_terminal",
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a shared learned adjacent-subchain pair terminal.")
    parser.add_argument("--out", default="reports/subchain_pair_terminal")
    parser.add_argument("--teacher-mode", choices=["static_recon", "recon_policy_terminal"], default="recon_policy_terminal")
    parser.add_argument("--policy-terminal-path", default="")
    parser.add_argument("--policy-terminal-blend", type=float, default=1.0)
    parser.add_argument("--policy-terminal-scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--policy-terminal-observation-mode", default="normalized_raw")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=8_000_000)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--no-pair-position", action="store_true")
    parser.add_argument("--theta-scale", type=float, default=0.21)
    parser.add_argument("--velocity-scale", type=float, default=5.0)
    parser.add_argument("--x-scale", type=float, default=2.4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-seed", type=int, default=8101)
    parser.add_argument("--subchain-blend", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-pair-pressure", type=float, default=0.02)
    parser.add_argument("--max-force-fraction", type=float, default=1.0)
    parser.add_argument("--confidence-boost", type=float, default=0.08)
    parser.add_argument("--urgency-boost", type=float, default=0.15)
    parser.add_argument("--eval-seed-start", type=int, default=1_900_000)
    parser.add_argument("--eval-seed-starts", type=int, nargs="*", default=[])
    parser.add_argument("--eval-episodes", type=int, default=20)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "out": report["config"]["out"],
                "checkpoint_path": report["checkpoint_path"],
                "base_success": report["base_eval"].get("success_rate", 0.0),
                "learned_success": report["learned_subchain_eval"].get("success_rate", 0.0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
