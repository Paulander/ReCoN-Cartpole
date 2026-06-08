from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import yaml

from recon_lite.plasticity import snapshot_bandit
from recon_cartpole.control.scripts import ProposalGains, gain_bounds
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.engine_runner import ReConCartPoleController, RunnerConfig
from recon_cartpole.recon.trace_db import graph_to_trace, save_trace
from recon_cartpole.training.evaluate import rollout
from recon_cartpole.visualization.physics_render import render_trace_html


def run_curriculum(path: str, output_dir: str | None = None) -> list[dict[str, Any]]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out_dir = Path(output_dir or "reports/curriculum_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = config.get("mode", "recon_fast_bandit")
    horizon = int(config.get("horizon", 500))
    train_start = int(config.get("seed_split", {}).get("train_start", 30_000))
    validation_start = int(config.get("seed_split", {}).get("validation_start", 130_000))
    rng = random.Random(int(config.get("seed", 7)))

    current_gains = ProposalGains.from_dict(config.get("initial_gains"))
    results: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(config.get("stages", [])):
        stage_dir = out_dir / f"{stage_index:02d}_{stage.get('name', 'stage')}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        result, current_gains = run_stage(
            stage=stage,
            mode=mode,
            horizon=horizon,
            train_seed=train_start + stage_index * 20_000,
            validation_seed=validation_start + stage_index * 20_000,
            starting_gains=current_gains,
            rng=rng,
            out_dir=stage_dir,
        )
        results.append(result)
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        write_curriculum_summary(results, out_dir)
    return results


def run_stage(
    stage: dict[str, Any],
    mode: str,
    horizon: int,
    train_seed: int,
    validation_seed: int,
    starting_gains: ProposalGains,
    rng: random.Random,
    out_dir: Path,
) -> tuple[dict[str, Any], ProposalGains]:
    n_poles = int(stage.get("n_poles", 1))
    train_episodes = int(stage.get("train_episodes", 100))
    eval_episodes = int(stage.get("eval_episodes", 25))
    gain_trials = int(stage.get("gain_trials", max(6, min(30, train_episodes // 100))))
    trial_train_episodes = max(1, train_episodes // max(1, gain_trials))

    best_score = -math.inf
    best_result: dict[str, Any] | None = None
    best_gains = starting_gains
    best_controller: ReConCartPoleController | None = None
    trials: list[dict[str, Any]] = []

    for trial_index in range(gain_trials):
        gains = starting_gains if trial_index == 0 else mutate_gains(best_gains, rng, float(stage.get("gain_sigma", 0.18)))
        controller = ReConCartPoleController(
            RunnerConfig(
                n_poles=n_poles,
                mode=mode,
                reset_bandit_each_episode=False,
                learn=True,
                stage=stage.get("name", "stage"),
                proposal_gains=gains,
            )
        )
        train_results = run_episodes(
            stage,
            controller,
            episodes=trial_train_episodes,
            seed=train_seed + trial_index * 1_000,
            horizon=horizon,
            trace=False,
        )
        controller.config.learn = False
        eval_results = run_episodes(
            stage,
            controller,
            episodes=eval_episodes,
            seed=validation_seed + trial_index * 1_000,
            horizon=horizon,
            trace=False,
        )
        train_steps = [float(item["steps"]) for item in train_results]
        eval_steps = [float(item["steps"]) for item in eval_results]
        score = score_eval(eval_steps, horizon)
        trial = {
            "trial": trial_index,
            "score": score,
            "train_steps": summarize(train_steps),
            "eval_steps": summarize(eval_steps),
            "gains": gains.to_dict(),
        }
        trials.append(trial)
        if score > best_score:
            best_score = score
            best_gains = gains
            best_controller = controller
            best_result = trial

    assert best_controller is not None and best_result is not None
    trace = rollout(
        make_env(stage, horizon),
        best_controller,
        seed=validation_seed + 999_999,
        horizon=horizon,
        trace=True,
    )
    metadata = {
        "stage": stage.get("name", "stage"),
        "n_poles": n_poles,
        "mode": mode,
        "horizon": horizon,
        "gains": best_gains.to_dict(),
        "graph": graph_to_trace(best_controller.graph),
    }
    save_trace(str(out_dir / "best_trace.json"), metadata, trace["trace"])
    render_trace_html(
        {"metadata": metadata, "steps": trace["trace"]},
        str(out_dir / "best_replay.html"),
        f"{stage.get('name', 'stage')} ReCoN replay",
    )

    stage_result = {
        "stage": stage.get("name", "stage"),
        "n_poles": n_poles,
        "mode": mode,
        "train_episodes_budget": train_episodes,
        "gain_trials": gain_trials,
        "trial_train_episodes": trial_train_episodes,
        "eval_episodes": eval_episodes,
        "best_score": best_score,
        "best_trial": best_result,
        "passed": passes_gate(best_result["eval_steps"], stage.get("pass", {})),
        "bandit": snapshot_bandit(best_controller.bandit_state),
        "trace_steps": int(trace["steps"]),
        "trace_return": float(trace["return"]),
        "best_gains": best_gains.to_dict(),
        "trials": trials,
        "report_dir": str(out_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(stage_result, indent=2), encoding="utf-8")
    return stage_result, best_gains


def run_episodes(
    stage: dict[str, Any],
    controller: ReConCartPoleController,
    episodes: int,
    seed: int,
    horizon: int,
    trace: bool,
) -> list[dict[str, Any]]:
    env = make_env(stage, horizon)
    return [rollout(env, controller, seed=seed + episode, horizon=horizon, trace=trace) for episode in range(episodes)]


def make_env(stage: dict[str, Any], horizon: int) -> CartPoleNEnv:
    return CartPoleNEnv(
        CartPoleNConfig(
            n_poles=int(stage.get("n_poles", 1)),
            horizon=horizon,
            initial_angle_range=float(stage.get("initial_angle_range", 0.05)),
            force_noise=float(stage.get("force_noise", 0.0)),
            damping=float(stage.get("damping", 0.01)),
        ),
        render_mode="rgb_array",
    )


def mutate_gains(gains: ProposalGains, rng: random.Random, sigma: float) -> ProposalGains:
    bounds = gain_bounds()
    values = asdict(gains)
    mutated = {}
    for key, value in values.items():
        low, high = bounds[key]
        factor = math.exp(rng.gauss(0.0, sigma))
        mutated[key] = max(low, min(high, float(value) * factor))
    return ProposalGains(**mutated).clipped()


def score_eval(steps: list[float], horizon: int) -> float:
    if not steps:
        return 0.0
    mean_steps = float(mean(steps))
    p10_steps = float(np.percentile(steps, 10))
    success = float(np.mean(np.asarray(steps) >= horizon))
    return mean_steps + 0.25 * p10_steps + 50.0 * success


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "max": float(max(values)),
    }


def passes_gate(eval_summary: dict[str, float], gate: dict[str, Any]) -> bool:
    if not gate:
        return False
    mean_gate = gate.get("mean_survival_steps", gate.get("mean_return"))
    p10_gate = gate.get("p10_survival_steps")
    if mean_gate is not None and eval_summary.get("mean", 0.0) < float(mean_gate):
        return False
    if p10_gate is not None and eval_summary.get("p10", 0.0) < float(p10_gate):
        return False
    return True


def write_curriculum_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    lines = [
        "# Curriculum Run",
        "",
        "This run trains shared proposal gains plus persistent bandit state per stage, then freezes learning for held-out evaluation.",
        "",
        "| stage | N | trials | eval mean | eval p10 | eval max | passed | report |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        eval_steps = result["best_trial"]["eval_steps"]
        lines.append(
            "| {stage} | {n} | {trials} | {mean:.1f} | {p10:.1f} | {maxv:.1f} | {passed} | [{label}]({href}/best_replay.html) |".format(
                stage=result["stage"],
                n=result["n_poles"],
                trials=result["gain_trials"],
                mean=eval_steps["mean"],
                p10=eval_steps["p10"],
                maxv=eval_steps["max"],
                passed="yes" if result["passed"] else "no",
                label="replay",
                href=Path(result["report_dir"]).name,
            )
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def save_curriculum_results(results: list[dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
