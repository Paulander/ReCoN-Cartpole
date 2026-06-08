from __future__ import annotations

import argparse
import json
from pathlib import Path

from recon_cartpole.training.curriculum import run_curriculum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curriculum_3_to_5.yaml")
    parser.add_argument("--out", default="reports/curriculum_3_to_5")
    args = parser.parse_args()
    results = run_curriculum(args.config, args.out)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        eval_steps = result["best_trial"]["eval_steps"]
        print(
            f"{result['stage']}: mean={eval_steps['mean']:.1f} "
            f"p10={eval_steps['p10']:.1f} passed={result['passed']}"
        )


if __name__ == "__main__":
    main()
