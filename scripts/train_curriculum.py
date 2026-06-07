from __future__ import annotations

import argparse

from recon_cartpole.training.curriculum import run_curriculum, save_curriculum_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curriculum_1_to_6.yaml")
    parser.add_argument("--out", default="reports/curriculum_results.json")
    args = parser.parse_args()
    results = run_curriculum(args.config)
    save_curriculum_results(results, args.out)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()

