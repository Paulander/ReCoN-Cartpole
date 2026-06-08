from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from recon_cartpole.training.ablations import run_ablations
from recon_cartpole.training.ppo_baseline import PPOBaselineConfig, run_ppo_baseline

DEFAULT_RECON_MODES = [
    "baseline_heuristic",
    "static_recon",
    "recon_fast",
    "recon_bandit",
    "recon_fast_bandit",
    "recon_slow",
    "recon_learn_only",
    "recon_slow_no_gain_search",
    "gain_search_only",
    "gain_search_recon_fast_bandit",
]


def run_nway_comparison(
    n_values: list[int],
    horizon: int,
    eval_episodes: int,
    seed_start: int,
    train_episodes: int,
    ppo_timesteps: int,
    out_dir: str | Path,
    env_params_by_n: dict[int, dict[str, Any]] | None = None,
    modes: list[str] | None = None,
    include_ppo: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "horizon": horizon,
        "eval_episodes": eval_episodes,
        "seed_start": seed_start,
        "train_episodes": train_episodes,
        "ppo_timesteps": ppo_timesteps,
        "n_values": n_values,
        "runs": [],
    }
    env_params_by_n = env_params_by_n or {}
    selected_modes = list(modes or DEFAULT_RECON_MODES)
    for n_poles in n_values:
        seeds = [seed_start + n_poles * 10_000 + idx for idx in range(eval_episodes)]
        env_params = dict(env_params_by_n.get(n_poles, {}))
        recon_results = run_ablations(
            n_poles=n_poles,
            horizon=horizon,
            seeds=seeds,
            modes=selected_modes,
            env_params=env_params,
            train_episodes=train_episodes,
            include_gain_search=False,
        )
        for row in recon_results:
            row["family"] = "recon"
            row["status"] = "completed"
            results["runs"].append(row)
        if include_ppo:
            ppo_row = run_ppo_baseline(
                PPOBaselineConfig(
                    n_poles=n_poles,
                    horizon=horizon,
                    train_timesteps=ppo_timesteps,
                    train_seed=seed_start + n_poles * 20_000,
                    eval_seeds=seeds,
                    env_params=env_params,
                )
            )
            ppo_row["family"] = "ppo"
            results["runs"].append(ppo_row)
    results["wall_clock_seconds"] = time.perf_counter() - started
    (out / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_comparison_markdown(results, out / "comparison.md")
    return results


def _mechanism_label(row: dict[str, Any]) -> str:
    return ", ".join(key for key, active in row.get("mechanisms", {}).items() if active) or "none"


def write_comparison_markdown(results: dict[str, Any], path: Path) -> None:
    lines = [
        "# ReCoN vs PPO Same-Seed Comparison",
        "",
        "Rows use identical held-out seeds and environment parameters within each N. PPO is explicit: if optional RL dependencies are missing, the PPO row is marked `unavailable` rather than omitted.",
        "",
        f"Horizon: `{results['horizon']}`",
        f"Eval episodes per N: `{results['eval_episodes']}`",
        f"ReCoN train episodes per mode: `{results['train_episodes']}`",
        f"PPO train timesteps: `{results['ppo_timesteps']}`",
        "",
        "| N | family | mode | status | mechanisms | mean | p10 | success | max |",
        "|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in results.get("runs", []):
        lines.append(
            "| {n} | {family} | {mode} | {status} | {mechanisms} | {mean:.1f} | {p10:.1f} | {success:.2f} | {maxv:.1f} |".format(
                n=row.get("n_poles"),
                family=row.get("family", ""),
                mode=row.get("mode", ""),
                status=row.get("status", "completed"),
                mechanisms=_mechanism_label(row),
                mean=float(row.get("mean_survival", 0.0)),
                p10=float(row.get("p10_survival", 0.0)),
                success=float(row.get("success_rate", 0.0)),
                maxv=float(row.get("max_survival", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "This is a comparison artifact, not a solved claim. A mode is solved only if it meets the configured held-out threshold with the required evaluation episode count.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
