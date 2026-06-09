from __future__ import annotations

import argparse
import json

from recon_cartpole.training.comparison import run_nway_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=260_000)
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--ppo-timesteps", type=int, default=50_000)
    parser.add_argument("--out", default="reports/recon_vs_ppo_comparison")
    parser.add_argument("--ppo-device", default="cpu")
    parser.add_argument("--ppo-n-envs", type=int, default=1)
    parser.add_argument("--no-ppo", action="store_true")
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--discrete-action-bins", type=int, default=2)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="parallel")
    args = parser.parse_args()
    env_params_by_n = {
        n: {
            "action_mode": args.action_mode,
            "dt": args.dt,
            "initial_angle_range": args.initial_angle_range,
            "force_noise": args.force_noise,
            "link_coupling": args.link_coupling,
            "force_mag": args.force_mag,
            "discrete_action_bins": args.discrete_action_bins,
            "dynamics_mode": args.dynamics_mode,
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
        ppo_device=args.ppo_device,
        ppo_n_envs=args.ppo_n_envs,
        env_params_by_n=env_params_by_n,
        include_ppo=not args.no_ppo,
    )
    print(json.dumps({"out": args.out, "runs": len(results["runs"]), "wall_clock_seconds": results["wall_clock_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
