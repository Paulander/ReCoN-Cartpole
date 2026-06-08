from __future__ import annotations

import argparse
import json

from recon_cartpole.training.comparison import run_nway_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=260_000)
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--ppo-timesteps", type=int, default=50_000)
    parser.add_argument("--out", default="reports/recon_vs_ppo_comparison")
    parser.add_argument("--no-ppo", action="store_true")
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    args = parser.parse_args()
    env_params_by_n = {
        n: {
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "link_coupling": args.link_coupling,
        }
        for n in args.n_values
    }
    results = run_nway_comparison(
        n_values=args.n_values,
        horizon=args.horizon,
        eval_episodes=args.eval_episodes,
        seed_start=args.seed_start,
        train_episodes=args.train_episodes,
        ppo_timesteps=args.ppo_timesteps,
        out_dir=args.out,
        env_params_by_n=env_params_by_n,
        include_ppo=not args.no_ppo,
    )
    print(json.dumps({"out": args.out, "runs": len(results["runs"]), "wall_clock_seconds": results["wall_clock_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
