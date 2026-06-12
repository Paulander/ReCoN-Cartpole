from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.mingru_terminal import MinGRUTerminal, MinGRUTerminalConfig
from recon_cartpole.training.ablations import summarize_steps
from recon_cartpole.training.evaluate import rollout

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def config_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


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


def terminal_config(args: argparse.Namespace, checkpoint_path: str, hidden: int, seq_len: int) -> MinGRUTerminalConfig:
    return MinGRUTerminalConfig(
        enabled=True,
        hidden_size=hidden,
        sequence_length=seq_len,
        observation_mode=args.observation_mode,
        include_prev_force=args.include_prev_force,
        include_context=args.include_context,
        blend=args.blend,
        scope=args.scope,
        confidence_floor=args.confidence_floor,
        checkpoint_path=checkpoint_path,
    )


def ladder_validation_seeds(args: argparse.Namespace) -> list[int]:
    starts = getattr(args, "validation_seed_starts", None) or [args.validation_seed_start]
    seeds: list[int] = []
    for start in starts:
        seeds.extend(int(start) + idx for idx in range(int(args.validation_episodes)))
    return seeds


def evaluate_recon_mingru(checkpoint_path: str, args: argparse.Namespace, seeds: list[int], hidden: int, seq_len: int) -> dict[str, Any]:
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=args.n_poles,
            mode="recon_mingru_terminal",
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            selection_mode=args.selection_mode,
            learn=False,
            mingru_terminal=terminal_config(args, checkpoint_path, hidden, seq_len),
        )
    )
    steps = []
    returns = []
    for seed in seeds:
        result = rollout(make_env(args), controller, seed=seed, horizon=args.horizon, trace=False)
        steps.append(float(result["steps"]))
        returns.append(float(result["return"]))
    summary = summarize_steps(steps, args.horizon)
    summary.update({"returns_mean": float(np.mean(returns)) if returns else 0.0, "episodes": len(seeds)})
    return summary


def evaluate_pure_mingru(checkpoint_path: str, args: argparse.Namespace, seeds: list[int], hidden: int, seq_len: int) -> dict[str, Any]:
    terminal = MinGRUTerminal(
        args.n_poles,
        args.force_mag,
        args.discrete_action_bins,
        terminal_config(args, checkpoint_path, hidden, seq_len),
    )
    steps = []
    returns = []
    for seed in seeds:
        env = make_env(args)
        obs, info = env.reset(seed=seed)
        terminal.reset()
        total = 0.0
        for step in range(args.horizon):
            prediction = terminal.predict(obs, info.get("raw_state"), {})
            force = 0.0 if prediction.force is None else float(prediction.force)
            action = action_from_force(force, "discrete", args.force_mag, args.discrete_action_bins)
            obs, reward, terminated, truncated, info = env.step(action)
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


def train_supervised_candidate(args: argparse.Namespace, hidden: int, seq_len: int, out: Path) -> str:
    from train_mingru_supervised import train

    report = train(
        SimpleNamespace(
            dataset=args.supervised_dataset,
            out=str(out),
            n_poles=args.n_poles,
            horizon=args.horizon,
            force_mag=args.force_mag,
            discrete_action_bins=args.discrete_action_bins,
            observation_mode=args.observation_mode,
            hidden_size=hidden,
            sequence_length=seq_len,
            include_prev_force=args.include_prev_force,
            include_context=args.include_context,
            scope=args.scope,
            blend=args.blend,
            confidence_floor=args.confidence_floor,
            epochs=args.supervised_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=0.1,
            value_weight=0.05,
            failure_weight=0.10,
            confidence_weight=0.05,
            min_sample_episode_survival=args.min_sample_episode_survival,
            max_sample_episode_survival=args.max_sample_episode_survival,
            failure_sample_weight=args.failure_sample_weight,
            late_sample_weight=args.late_sample_weight,
            low_return_sample_weight=args.low_return_sample_weight,
            max_grad_norm=1.0,
            device=args.device,
            seed=args.train_seed,
        )
    )
    return str(report["checkpoint_path"])


def candidate_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    checkpoints = [item for item in args.checkpoints if item]
    specs: list[dict[str, Any]] = []
    for hidden in args.hidden_sizes:
        for seq_len in args.sequence_lengths:
            if checkpoints:
                for checkpoint in checkpoints:
                    specs.append({"hidden": hidden, "sequence_length": seq_len, "checkpoint": checkpoint})
            elif args.supervised_dataset:
                specs.append({"hidden": hidden, "sequence_length": seq_len, "checkpoint": ""})
    return specs


def run_ladder(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = ladder_validation_seeds(args)
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(candidate_specs(args)):
        c_payload = {
            "env": {
                "n_poles": args.n_poles,
                "horizon": args.horizon,
                "dt": args.dt,
                "dynamics_mode": args.dynamics_mode,
                "discrete_action_bins": args.discrete_action_bins,
                "force_mag": args.force_mag,
                "initial_angle_range": args.initial_angle_range,
                "force_noise": args.force_noise,
                "link_coupling": args.link_coupling,
            },
            "terminal": spec,
            "scope": args.scope,
            "selection_mode": args.selection_mode,
            "observation_mode": args.observation_mode,
        }
        cid = config_hash(c_payload)
        candidate_out = out / f"candidate_{idx:02d}_{cid}"
        candidate_out.mkdir(parents=True, exist_ok=True)
        checkpoint = spec["checkpoint"] or train_supervised_candidate(
            args, int(spec["hidden"]), int(spec["sequence_length"]), candidate_out
        )
        pure = evaluate_pure_mingru(
            checkpoint, args, seeds, int(spec["hidden"]), int(spec["sequence_length"])
        )
        recon = evaluate_recon_mingru(
            checkpoint, args, seeds, int(spec["hidden"]), int(spec["sequence_length"])
        )
        keep = bool(recon["mean_survival"] >= args.min_mean_gate or recon["success_rate"] >= args.min_success_gate)
        row = {
            "candidate_id": cid,
            "checkpoint_path": checkpoint,
            "hidden_size": int(spec["hidden"]),
            "sequence_length": int(spec["sequence_length"]),
            "pure_mingru_policy": pure,
            "recon_mingru_terminal": recon,
            "kept": keep,
            "mechanisms": {
                "minGRU_terminal": True,
                "pure_policy_baseline": True,
                "feedforward_policy_terminal": False,
                "edge_plasticity": False,
                "bandit_persistence": False,
                "slow_consolidation": False,
                "gain_mutation": False,
            },
        }
        rows.append(row)
        (candidate_out / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    rows.sort(key=lambda item: item["recon_mingru_terminal"]["mean_survival"], reverse=True)
    report = {
        "status": "completed" if rows else "no_candidates",
        "rows": rows[: args.keep_top_k],
        "all_rows": rows,
        "validation_seeds": seeds,
        "claim_discipline": "No solved claim is made by this ladder; run final held-out seeds once thresholds pass.",
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "leaderboard.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out / "leaderboard.md")
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Recurrent Terminal Ladder",
        "",
        report["claim_discipline"],
        "",
        "| candidate | kept | checkpoint | pure mean | pure p10 | pure success | ReCoN mean | ReCoN p10 | ReCoN success |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["all_rows"]:
        pure = row["pure_mingru_policy"]
        recon = row["recon_mingru_terminal"]
        lines.append(
            f"| {row['candidate_id']} | {row['kept']} | `{row['checkpoint_path']}` | "
            f"{pure['mean_survival']:.1f} | {pure['p10_survival']:.1f} | "
            f"{pure['success_rate']:.2f} | {recon['mean_survival']:.1f} | "
            f"{recon['p10_survival']:.1f} | {recon['success_rate']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--supervised-dataset", default="")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force"], default="normalized_raw")
    parser.add_argument("--scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--confidence-floor", type=float, default=0.05)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--include-prev-force", action="store_true", default=True)
    parser.add_argument("--no-prev-force", dest="include_prev_force", action="store_false")
    parser.add_argument("--include-context", action="store_true", default=True)
    parser.add_argument("--no-context", dest="include_context", action="store_false")
    parser.add_argument("--supervised-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-sample-episode-survival", type=float, default=0.0)
    parser.add_argument("--max-sample-episode-survival", type=float, default=0.0)
    parser.add_argument("--failure-sample-weight", type=float, default=0.0)
    parser.add_argument("--late-sample-weight", type=float, default=0.0)
    parser.add_argument("--low-return-sample-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-seed", type=int, default=9117)
    parser.add_argument("--validation-seed-start", type=int, default=820000)
    parser.add_argument("--validation-seed-starts", type=int, nargs="+", default=None)
    parser.add_argument("--validation-episodes", type=int, default=60)
    parser.add_argument("--min-mean-gate", type=float, default=250.0)
    parser.add_argument("--min-success-gate", type=float, default=0.05)
    parser.add_argument("--keep-top-k", type=int, default=3)
    parser.add_argument("--out", default="reports/recurrent_terminal_ladder")
    args = parser.parse_args()
    report = run_ladder(args)
    print(json.dumps({"status": report["status"], "out": args.out, "candidates": len(report["all_rows"])}, indent=2))


if __name__ == "__main__":
    main()
