from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from recon_lite.plasticity import ConsolidationConfig, PlasticityConfig
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.trace_db import graph_to_trace, save_trace
from recon_cartpole.training.evaluate import rollout
from recon_cartpole.visualization.physics_render import render_trace_html
from recon_cartpole.recon.node_params import NodeParamConfig
from recon_cartpole.recon.mlp_terminal import MlpTerminalConfig


@dataclass
class SolveThreshold:
    mean_survival: float
    p10_survival: float
    success_rate: float
    eval_episodes: int = 300
    horizon: int = 500


@dataclass
class IterationConfig:
    n_poles: int
    mode: str = "recon_learn_only"
    action_mode: str = "discrete"
    horizon: int = 500
    budget_episodes: int = 50_000
    train_block_episodes: int = 250
    eval_episodes: int = 50
    seed: int = 300_000
    validation_seed: int = 430_000
    initial_angle_range: float = 0.05
    force_noise: float = 0.02
    link_coupling: float = 12.0
    force_mag: float = 10.0
    discrete_action_bins: int = 2
    out_dir: str = "reports/train_until_solved"
    target: str = "auto"
    selection_mode: str = "soft_select"
    resume_checkpoint: str | None = None
    plasticity_eta: float = 0.03
    node_eta: float = 0.01
    consolidation_eta: float = 0.02
    consolidation_min_episodes: int = 20
    mlp_eta: float = 0.08
    mlp_eta_tick: float = 0.01
    mlp_sigma: float = 0.08
    mlp_blend: float = 0.35
    mlp_hidden_size: int = 16


@dataclass
class FailureTaxonomy:
    rail_left: int = 0
    rail_right: int = 0
    pole_failed: dict[str, int] = field(default_factory=dict)
    high_angular_velocity: int = 0
    bad_force_sign: int = 0
    oscillatory_force: int = 0
    insufficient_cart_recentering: int = 0
    overfocus_on_one_pole: int = 0
    solved_or_truncated: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def target_threshold(target: str, n_poles: int, eval_episodes: int, horizon: int) -> SolveThreshold:
    if target == "solved_n3" or (target == "auto" and n_poles == 3):
        return SolveThreshold(475.0, 400.0, 0.80, max(eval_episodes, 300), horizon)
    if target == "solved_n4" or (target == "auto" and n_poles == 4):
        return SolveThreshold(475.0, 350.0, 0.70, max(eval_episodes, 300), horizon)
    return SolveThreshold(475.0, 350.0, 0.80, max(eval_episodes, 300), horizon)


def make_env(config: IterationConfig) -> CartPoleNEnv:
    return CartPoleNEnv(
        CartPoleNConfig(
            n_poles=config.n_poles,
            action_mode=config.action_mode,
            horizon=config.horizon,
            initial_angle_range=config.initial_angle_range,
            force_noise=config.force_noise,
            link_coupling=config.link_coupling,
            force_mag=config.force_mag,
            discrete_action_bins=config.discrete_action_bins,
        ),
        render_mode="rgb_array",
    )


def make_controller(config: IterationConfig) -> ReConCartPoleController:
    controller = ReConCartPoleController(
        RunnerConfig(
            n_poles=config.n_poles,
            mode=config.mode,
            action_mode=config.action_mode,
            learn=True,
            reset_bandit_each_episode=False,
            stage=f"train_until_solved_n{config.n_poles}",
            selection_mode=config.selection_mode,
            force_mag=config.force_mag,
            discrete_action_bins=config.discrete_action_bins,
            plasticity=PlasticityConfig(eta_tick=config.plasticity_eta),
            node_params=NodeParamConfig(
                enabled=config.mode in ("recon_learn_only", "recon_slow_no_gain_search", "recon_mlp_terminal"),
                eta_fast=config.node_eta,
                eta_consolidate=config.consolidation_eta,
                min_episodes=config.consolidation_min_episodes,
            ),
            consolidation=ConsolidationConfig(
                enabled=config.mode in ("recon_slow", "recon_learn_only", "recon_slow_no_gain_search", "recon_mlp_terminal"),
                eta_consolidate=config.consolidation_eta,
                min_episodes=config.consolidation_min_episodes,
            ),
            mlp_terminal=MlpTerminalConfig(
                enabled=config.mode == "recon_mlp_terminal",
                hidden_size=config.mlp_hidden_size,
                eta=config.mlp_eta,
                eta_tick=config.mlp_eta_tick,
                sigma=config.mlp_sigma,
                blend=config.mlp_blend,
            ),
        )
    )
    if config.resume_checkpoint:
        controller.load_consolidation_checkpoint(config.resume_checkpoint)
    return controller


def summarize(steps: list[float], horizon: int) -> dict[str, float]:
    values = np.asarray(steps, dtype=float)
    if values.size == 0:
        return {"mean_survival": 0.0, "p10_survival": 0.0, "success_rate": 0.0, "max_survival": 0.0}
    return {
        "mean_survival": float(mean(values)),
        "p10_survival": float(np.percentile(values, 10)),
        "success_rate": float(np.mean(values >= horizon)),
        "max_survival": float(np.max(values)),
    }


def passes_threshold(summary: dict[str, float], threshold: SolveThreshold) -> bool:
    return (
        summary["mean_survival"] >= threshold.mean_survival
        and summary["p10_survival"] >= threshold.p10_survival
        and summary["success_rate"] >= threshold.success_rate
    )


def classify_failure(trace_steps: list[dict[str, Any]], n_poles: int, horizon: int) -> dict[str, Any]:
    taxonomy = FailureTaxonomy()
    if not trace_steps:
        return taxonomy.to_dict()
    final = trace_steps[-1]
    raw = final.get("raw_state", [])
    if final.get("truncated") and final.get("step", 0) + 1 >= horizon:
        taxonomy.solved_or_truncated += 1
    if raw:
        x = float(raw[0])
        if x < -2.2:
            taxonomy.rail_left += 1
        if x > 2.2:
            taxonomy.rail_right += 1
        theta = raw[2 : 2 + n_poles]
        theta_dot = raw[2 + n_poles : 2 + 2 * n_poles]
        for idx, angle in enumerate(theta):
            if abs(float(angle)) > 0.20:
                key = f"pole_{idx}"
                taxonomy.pole_failed[key] = taxonomy.pole_failed.get(key, 0) + 1
        if any(abs(float(value)) > 3.0 for value in theta_dot):
            taxonomy.high_angular_velocity += 1
        if abs(x) > 1.4 and np.sign(final.get("force", 0.0)) == np.sign(x):
            taxonomy.insufficient_cart_recentering += 1
    forces = [float(step.get("force", 0.0)) for step in trace_steps[-20:]]
    if len(forces) >= 6:
        sign_flips = sum(1 for a, b in zip(forces, forces[1:]) if np.sign(a) != np.sign(b))
        if sign_flips > len(forces) * 0.55:
            taxonomy.oscillatory_force += 1
    regimes = [step.get("selected_regime") for step in trace_steps[-30:] if step.get("selected_regime")]
    if regimes:
        top_count = max(regimes.count(regime) for regime in set(regimes))
        if top_count >= max(10, int(0.85 * len(regimes))):
            taxonomy.overfocus_on_one_pole += 1
    return taxonomy.to_dict()


def merge_taxonomy(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"pole_failed": {}}
    for item in items:
        for key, value in item.items():
            if key == "pole_failed":
                for pole, count in value.items():
                    merged["pole_failed"][pole] = merged["pole_failed"].get(pole, 0) + count
            else:
                merged[key] = merged.get(key, 0) + int(value)
    return merged


def adapt_config(config: IterationConfig, taxonomy: dict[str, Any]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    if taxonomy.get("oscillatory_force", 0) > 0:
        config.node_eta *= 0.8
        config.plasticity_eta *= 0.85
        changes["dampen_learning_rates"] = {"node_eta": config.node_eta, "plasticity_eta": config.plasticity_eta}
    if taxonomy.get("insufficient_cart_recentering", 0) > 0 or taxonomy.get("rail_left", 0) + taxonomy.get("rail_right", 0) > 0:
        config.consolidation_eta *= 1.1
        changes["increase_consolidation_eta"] = config.consolidation_eta
    if taxonomy.get("high_angular_velocity", 0) > 0:
        config.node_eta *= 1.05
        changes["increase_node_eta"] = config.node_eta
    if not changes:
        config.plasticity_eta *= 1.02
        changes["small_plasticity_eta_probe"] = config.plasticity_eta
    return changes


def evaluate_controller(
    controller: ReConCartPoleController,
    config: IterationConfig,
    eval_episodes: int,
    seed_start: int,
    trace_failures: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    controller.config.learn = False
    steps: list[float] = []
    returns: list[float] = []
    taxonomies: list[dict[str, Any]] = []
    for episode in range(eval_episodes):
        env = make_env(config)
        trace = episode < trace_failures
        result = rollout(env, controller, seed=seed_start + episode, horizon=config.horizon, trace=trace)
        steps.append(float(result["steps"]))
        returns.append(float(result["return"]))
        if trace:
            taxonomies.append(classify_failure(result.get("trace", []), config.n_poles, config.horizon))
    summary = summarize(steps, config.horizon)
    summary.update({"returns_mean": float(mean(returns)) if returns else 0.0, "episodes": eval_episodes})
    controller.config.learn = True
    return summary, taxonomies


def run_train_until_solved(config: IterationConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = target_threshold(config.target, config.n_poles, config.eval_episodes, config.horizon)
    controller = make_controller(config)
    start_time = time.perf_counter()
    total_train_episodes = 0
    iteration = 0
    best_summary: dict[str, Any] | None = None
    best_score = -1.0
    history: list[dict[str, Any]] = []

    while total_train_episodes < config.budget_episodes:
        block = min(config.train_block_episodes, config.budget_episodes - total_train_episodes)
        train_steps: list[float] = []
        for episode in range(block):
            env = make_env(config)
            result = rollout(
                env,
                controller,
                seed=config.seed + total_train_episodes + episode,
                horizon=config.horizon,
                trace=False,
            )
            train_steps.append(float(result["steps"]))
        total_train_episodes += block
        eval_summary, taxonomies = evaluate_controller(
            controller,
            config,
            min(config.eval_episodes, threshold.eval_episodes),
            config.validation_seed,
        )
        taxonomy = merge_taxonomy(taxonomies)
        score = eval_summary["mean_survival"] + 0.25 * eval_summary["p10_survival"] + 50.0 * eval_summary["success_rate"]
        promoted = score > best_score
        if promoted:
            best_score = score
            best_summary = dict(eval_summary)
            controller.save_checkpoint(str(out_dir / "best_checkpoint.json"))
        solved = passes_threshold(eval_summary, threshold) and eval_summary["episodes"] >= threshold.eval_episodes
        changes = {} if solved else adapt_config(config, taxonomy)
        record = {
            "iteration": iteration,
            "train_episodes_total": total_train_episodes,
            "train_steps_mean": float(mean(train_steps)) if train_steps else 0.0,
            "eval": eval_summary,
            "failure_taxonomy": taxonomy,
            "config_changes": changes,
            "promoted": promoted,
            "solved": solved,
            "mechanisms": controller.learning_mechanisms(),
            "checkpoint": str(out_dir / "best_checkpoint.json") if promoted else None,
        }
        history.append(record)
        (out_dir / "iterations.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if solved:
            break
        iteration += 1

    controller.config.learn = False
    trace_result = rollout(make_env(config), controller, seed=config.validation_seed + 999_999, horizon=config.horizon, trace=True)
    metadata = {
        "env": "CartPoleN",
        "n_poles": config.n_poles,
        "mode": config.mode,
        "mechanisms": controller.learning_mechanisms(),
        "train_until_solved_config": asdict(config),
        "best_summary": best_summary,
        "graph": graph_to_trace(controller.graph),
    }
    save_trace(str(out_dir / "final_trace.json"), metadata, trace_result["trace"])
    render_trace_html({"metadata": metadata, "steps": trace_result["trace"]}, str(out_dir / "final_replay.html"), f"Train-until-solved N={config.n_poles}")
    report = {
        "status": "solved" if history and history[-1]["solved"] else "budget_exhausted",
        "threshold": asdict(threshold),
        "best_summary": best_summary,
        "train_episodes": total_train_episodes,
        "wall_clock_seconds": time.perf_counter() - start_time,
        "history": history,
        "final_trace": str(out_dir / "final_trace.json"),
        "final_replay": str(out_dir / "final_replay.html"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, out_dir)
    return report


def write_markdown_report(report: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Train Until Solved Report",
        "",
        f"Status: `{report['status']}`",
        f"Train episodes: `{report['train_episodes']}`",
        f"Wall-clock seconds: `{report['wall_clock_seconds']:.2f}`",
        "",
        "| iter | train episodes | eval mean | eval p10 | success | promoted | solved |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in report.get("history", []):
        eval_summary = item["eval"]
        lines.append(
            f"| {item['iteration']} | {item['train_episodes_total']} | {eval_summary['mean_survival']:.1f} | {eval_summary['p10_survival']:.1f} | {eval_summary['success_rate']:.2f} | {'yes' if item['promoted'] else 'no'} | {'yes' if item['solved'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Claim Discipline",
        "",
        "This report is a training attempt artifact. Do not claim solved unless status is `solved` and the held-out threshold block passed with the required episode count.",
    ])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
