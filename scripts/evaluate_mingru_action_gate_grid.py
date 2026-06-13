from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_mingru_action_gate import evaluate, eval_seeds  # noqa: E402


def _floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def candidate_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(row.get("success_rate", 0.0)),
        float(row.get("p10_survival", 0.0)),
        float(row.get("cvar_survival", 0.0)),
        float(row.get("mean_survival", 0.0)),
        -float(row.get("override_count", 0.0)),
    )


def candidate_configs(args: argparse.Namespace) -> list[tuple[float, float, float]]:
    configs = [
        (float(confidence), float(margin), float(apply_threshold))
        for confidence in _floats(args.gate_confidences)
        for margin in _floats(args.gate_margins)
        for apply_threshold in _floats(args.gate_apply_thresholds)
    ]
    max_candidates = int(getattr(args, "max_candidates", 0) or 0)
    if max_candidates > 0:
        return configs[:max_candidates]
    return configs


def make_eval_args(args: argparse.Namespace, confidence: float, margin: float, apply_threshold: float) -> argparse.Namespace:
    payload = vars(args).copy()
    payload["gate_confidence"] = float(confidence)
    payload["gate_margin"] = float(margin)
    payload["gate_apply_threshold"] = float(apply_threshold)
    return argparse.Namespace(**payload)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    base = result["base_eval"]
    lines = [
        "# minGRU Action-Gate Threshold Grid",
        "",
        f"Status: `{result['status']}`",
        f"Checkpoint: `{result['checkpoint_path']}`",
        f"Gate path: `{result['gate_path']}`",
        f"Episodes: `{base['episodes']}`",
        "",
        "| confidence | margin | apply | mean | p10 | cvar | success | overrides | delta mean | delta p10 | delta cvar |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.get("candidates", []):
        lines.append(
            f"| {row['gate_confidence']:.3f} | {row['gate_margin']:.3f} | {row['gate_apply_threshold']:.3f} | "
            f"{row['mean_survival']:.1f} | {row['p10_survival']:.1f} | {row['cvar_survival']:.1f} | "
            f"{row['success_rate']:.3f} | {row['override_count']} | {row['delta_mean_survival']:.2f} | "
            f"{row['delta_p10_survival']:.2f} | {row['delta_cvar_survival']:.2f} |"
        )
    best = result.get("best") or {}
    if best:
        lines.extend(
            [
                "",
                "## Best Candidate",
                "",
                f"Confidence `{best['gate_confidence']:.3f}`, margin `{best['gate_margin']:.3f}`, apply `{best['gate_apply_threshold']:.3f}`.",
                f"Success `{best['success_rate']:.4f}`, p10 `{best['p10_survival']:.1f}`, cvar `{best['cvar_survival']:.1f}`, overrides `{best['override_count']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "This is a held-out threshold sweep over a fixed learned gate. It is not a solve claim unless held-out metrics clear the configured solve threshold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_grid(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = eval_seeds(args)
    base_eval = evaluate(args, seeds)
    rows: list[dict[str, Any]] = []
    gate_path = Path(args.gate_path)
    configs = candidate_configs(args)
    for confidence, margin, apply_threshold in configs:
        eval_args = make_eval_args(args, confidence, margin, apply_threshold)
        row = evaluate(eval_args, seeds, gate_path)
        row.update(
            {
                "gate_confidence": float(confidence),
                "gate_margin": float(margin),
                "gate_apply_threshold": float(apply_threshold),
                "delta_mean_survival": float(row["mean_survival"] - base_eval["mean_survival"]),
                "delta_p10_survival": float(row["p10_survival"] - base_eval["p10_survival"]),
                "delta_cvar_survival": float(row["cvar_survival"] - base_eval["cvar_survival"]),
                "delta_success_rate": float(row["success_rate"] - base_eval["success_rate"]),
            }
        )
        rows.append(row)
        partial = {
            "status": "running",
            "checkpoint_path": args.checkpoint_path,
            "gate_path": args.gate_path,
            "base_eval": base_eval,
            "candidates": rows,
            "best": max(rows, key=candidate_key),
        }
        (out / "summary.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
        write_markdown(partial, out / "summary.md")
    best = max(rows, key=candidate_key) if rows else None
    result = {
        "status": "completed",
        "out": str(out),
        "checkpoint_path": args.checkpoint_path,
        "gate_path": args.gate_path,
        "seed_starts": args.eval_seed_starts or [args.eval_seed_start],
        "eval_episodes": int(args.eval_episodes),
        "max_candidates": int(getattr(args, "max_candidates", 0) or 0),
        "base_eval": base_eval,
        "candidates": rows,
        "best": best,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, out / "summary.md")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained minGRU action gate across decision thresholds.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--gate-path", required=True)
    parser.add_argument("--out", default="reports/mingru_action_gate_grid")
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
    parser.add_argument("--observation-mode", default="normalized_raw4_subchains_prev_force")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--include-prev-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-motif-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motif-model-path", default="")
    parser.add_argument("--motif-score-scale", type=float, default=10.0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--passthrough-confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-logit-margin-floor", type=float, default=0.0)
    parser.add_argument("--residual-feature-mode", choices=["basic", "proposal_diagnostics", "subchain_diagnostics"], default="subchain_diagnostics")
    parser.add_argument("--gate-confidences", default="0.75,0.85,0.92,0.97")
    parser.add_argument("--gate-margins", default="0.05,0.20,0.40")
    parser.add_argument("--gate-apply-thresholds", default="0.65,0.80,0.92")
    parser.add_argument("--gate-confidence", type=float, default=0.75)
    parser.add_argument("--gate-margin", type=float, default=0.05)
    parser.add_argument("--gate-apply-threshold", type=float, default=0.65)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--eval-seed-start", type=int, default=1900000)
    parser.add_argument("--eval-seed-starts", type=int, nargs="*", default=[])
    parser.add_argument("--eval-episodes", type=int, default=20)
    return parser


def main() -> None:
    result = run_grid(build_parser().parse_args())
    best = result.get("best") or {}
    print(
        json.dumps(
            {
                "out": result["out"],
                "status": result["status"],
                "base_success": result["base_eval"].get("success_rate", 0.0),
                "best_success": best.get("success_rate", 0.0),
                "best_p10": best.get("p10_survival", 0.0),
                "best_overrides": best.get("override_count", 0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
