from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recon_cartpole.training.train_until_solved import IterationConfig, run_train_until_solved


def run_dt_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = args.resume_checkpoint
    stages: list[dict[str, Any]] = []
    for index, dt in enumerate(args.dt_values):
        stage_dir = out_root / f"{index:02d}_dt_{dt:g}"
        config = IterationConfig(
            n_poles=args.n_poles,
            mode=args.mode,
            selection_mode=args.selection_mode,
            action_mode=args.action_mode,
            horizon=args.horizon,
            dt=dt,
            budget_episodes=args.budget_episodes,
            train_block_episodes=args.train_block_episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed + index * 100_000,
            validation_seed=args.validation_seed + index * 100_000,
            initial_angle_range=args.initial_angle_range,
            force_noise=args.force_noise,
            link_coupling=args.link_coupling,
            force_mag=args.force_mag,
            discrete_action_bins=args.discrete_action_bins,
            dynamics_mode=args.dynamics_mode,
            out_dir=str(stage_dir),
            resume_checkpoint=resume_checkpoint,
            mlp_eta=args.mlp_eta,
            mlp_eta_tick=args.mlp_eta_tick,
            mlp_sigma=args.mlp_sigma,
            mlp_blend=args.mlp_blend,
            mlp_hidden_size=args.mlp_hidden_size,
        )
        report = run_train_until_solved(config)
        best_checkpoint = stage_dir / "best_checkpoint.json"
        if best_checkpoint.exists():
            resume_checkpoint = str(best_checkpoint)
        stages.append(
            {
                "dt": dt,
                "status": report["status"],
                "best_summary": report.get("best_summary"),
                "train_episodes": report.get("train_episodes"),
                "report_dir": str(stage_dir),
                "resume_checkpoint": resume_checkpoint,
            }
        )
        (out_root / "summary.json").write_text(json.dumps({"stages": stages}, indent=2), encoding="utf-8")
        write_markdown(stages, out_root / "summary.md")
        if args.stop_on_unsolved and report["status"] != "solved":
            break
    return {"stages": stages}


def write_markdown(stages: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# ReCoN dt Curriculum",
        "",
        "Each row is a persisted training stage. A later stage resumes the previous stage's best checkpoint when available.",
        "",
        "| stage | dt | status | mean | p10 | success | train episodes | report |",
        "|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for index, stage in enumerate(stages):
        summary = stage.get("best_summary") or {}
        lines.append(
            "| {index} | {dt:g} | {status} | {mean:.1f} | {p10:.1f} | {success:.2f} | {episodes} | [{label}]({href}/report.md) |".format(
                index=index,
                dt=float(stage["dt"]),
                status=stage["status"],
                mean=float(summary.get("mean_survival", 0.0)),
                p10=float(summary.get("p10_survival", 0.0)),
                success=float(summary.get("success_rate", 0.0)),
                episodes=stage.get("train_episodes", 0),
                label=Path(stage["report_dir"]).name,
                href=Path(stage["report_dir"]).name,
            )
        )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "This is a curriculum artifact. A harder dt is not solved merely because an easier dt is solved.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--dt-values", type=float, nargs="+", required=True)
    parser.add_argument("--mode", default="recon_mlp_terminal")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="hard_select")
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="serial_lagrange")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--budget-episodes", type=int, default=300)
    parser.add_argument("--train-block-episodes", type=int, default=60)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--validation-seed", type=int, default=3_500_000)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--mlp-eta", type=float, default=0.02)
    parser.add_argument("--mlp-eta-tick", type=float, default=0.001)
    parser.add_argument("--mlp-sigma", type=float, default=0.02)
    parser.add_argument("--mlp-blend", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-size", type=int, default=16)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--stop-on-unsolved", action="store_true")
    parser.add_argument("--out", default="reports/dt_curriculum")
    args = parser.parse_args()
    result = run_dt_curriculum(args)
    print(json.dumps({"out": args.out, "stages": len(result["stages"])}, indent=2))


if __name__ == "__main__":
    main()
