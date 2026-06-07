from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from recon_lite.plasticity import snapshot_bandit
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.trace_db import graph_to_trace, save_trace
from recon_cartpole.training.evaluate import rollout
from recon_cartpole.visualization.physics_render import render_trace_html


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)) if values else 0.0,
        "median": float(median(values)) if values else 0.0,
        "p10": float(np.percentile(values, 10)) if values else 0.0,
        "p90": float(np.percentile(values, 90)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def run_for_link_count(
    n_links: int,
    mode: str,
    train_episodes: int,
    eval_episodes: int,
    horizon: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    env = CartPoleNEnv(CartPoleNConfig(n_poles=n_links, horizon=horizon), render_mode="rgb_array")
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=n_links,
            mode=mode,
            reset_bandit_each_episode=False,
            learn=True,
            stage=f"nlink_{n_links}_train",
        )
    )

    train_results = []
    for episode in range(train_episodes):
        result = rollout(env, controller, seed=seed + episode, horizon=horizon, trace=False)
        train_results.append(result)

    controller.config.learn = False
    eval_results = []
    eval_seed = seed + 100_000
    for episode in range(eval_episodes):
        result = rollout(env, controller, seed=eval_seed + episode, horizon=horizon, trace=False)
        eval_results.append(result)

    trace_result = rollout(env, controller, seed=eval_seed + eval_episodes, horizon=horizon, trace=True)
    metadata = {
        "env": "CartPoleN",
        "n_poles": n_links,
        "mode": mode,
        "horizon": horizon,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "graph": graph_to_trace(controller.graph),
    }
    trace_path = out_dir / f"nlink_{n_links}_trace.json"
    html_path = out_dir / f"nlink_{n_links}_replay.html"
    save_trace(str(trace_path), metadata, trace_result["trace"])
    render_trace_html(
        {"metadata": metadata, "steps": trace_result["trace"]},
        str(html_path),
        f"ReCoN N-link CartPole N={n_links}",
    )

    train_steps = [float(item["steps"]) for item in train_results]
    eval_steps = [float(item["steps"]) for item in eval_results]
    metrics = {
        "n_links": n_links,
        "mode": mode,
        "horizon": horizon,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "train_steps": summarize(train_steps),
        "train_last_10_mean": float(mean(train_steps[-10:])) if train_steps else 0.0,
        "eval_steps": summarize(eval_steps),
        "eval_success_rate_at_horizon": float(np.mean(np.asarray(eval_steps) >= horizon)),
        "trace_steps": int(trace_result["steps"]),
        "trace_return": float(trace_result["return"]),
        "bandit": snapshot_bandit(controller.bandit_state),
        "trace_json": str(trace_path),
        "trace_html": str(html_path),
    }
    return metrics


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    lines = [
        "# N-link 3/4/5 Training Sweep",
        "",
        "This run carries bandit priors across training episodes, freezes learning for held-out evaluation, and exports one replay per link count.",
        "The custom N-link dynamics are still a benchmark scaffold, so treat these as iteration metrics rather than solved-control claims.",
        "",
        "| links | mode | train episodes | eval episodes | eval mean steps | eval p10 | eval max | success@horizon | replay |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| {n_links} | {mode} | {train_episodes} | {eval_episodes} | {mean:.1f} | {p10:.1f} | {maxv:.1f} | {success:.2f} | [{label}]({href}) |".format(
                n_links=result["n_links"],
                mode=result["mode"],
                train_episodes=result["train_episodes"],
                eval_episodes=result["eval_episodes"],
                mean=result["eval_steps"]["mean"],
                p10=result["eval_steps"]["p10"],
                maxv=result["eval_steps"]["max"],
                success=result["eval_success_rate_at_horizon"],
                label=f"N={result['n_links']}",
                href=Path(result["trace_html"]).name,
            )
        )
    lines.append("")
    lines.append("## Next Control Work")
    lines.append("")
    lines.append("1. Validate or replace the approximate coupled N-link dynamics before making high-N claims.")
    lines.append("2. Promote slow consolidation after training/eval split is stable.")
    lines.append("3. Add failure taxonomy by rail exit, base-pole angle, and outer-link divergence.")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--mode", default="recon_fast_bandit")
    parser.add_argument("--train-episodes", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--out", default="reports/nlink_3_5_training")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for n_links in args.links:
        print(f"training n_links={n_links} mode={args.mode}")
        result = run_for_link_count(
            n_links=n_links,
            mode=args.mode,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            horizon=args.horizon,
            seed=args.seed + n_links * 10_000,
            out_dir=out_dir,
        )
        print(json.dumps(result, indent=2))
        results.append(result)

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results, out_dir)
    print(f"wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
