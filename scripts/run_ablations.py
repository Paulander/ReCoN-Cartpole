from __future__ import annotations

import argparse

from recon_cartpole.training.ablations import run_ablations, write_ablation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-poles", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=230_000)
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--include-gain-search", action="store_true")
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.0)
    parser.add_argument("--link-coupling", type=float, default=0.35)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--discrete-action-bins", type=int, default=2)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="parallel")
    parser.add_argument("--out", default="reports/ablations")
    args = parser.parse_args()
    seeds = [args.seed_start + idx for idx in range(args.episodes)]
    env_params = {
        "action_mode": args.action_mode,
        "initial_angle_range": args.initial_angle_range,
        "force_noise": args.force_noise,
        "link_coupling": args.link_coupling,
        "force_mag": args.force_mag,
        "discrete_action_bins": args.discrete_action_bins,
        "dynamics_mode": args.dynamics_mode,
    }
    results = run_ablations(
        n_poles=args.n_poles,
        horizon=args.horizon,
        seeds=seeds,
        env_params=env_params,
        train_episodes=args.train_episodes,
        include_gain_search=args.include_gain_search,
    )
    write_ablation_report(results, args.out)
    print(f"wrote {args.out}/ablations.md and {args.out}/ablations.json")


if __name__ == "__main__":
    main()
