from __future__ import annotations

import argparse
import json

from recon_cartpole.training.train_until_solved import IterationConfig, run_train_until_solved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-poles", type=int, required=True)
    parser.add_argument("--target", default="auto")
    parser.add_argument("--mode", default="recon_learn_only")
    parser.add_argument("--selection-mode", choices=["soft_select", "hard_select"], default="soft_select")
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--budget-episodes", type=int, default=50_000)
    parser.add_argument("--train-block-episodes", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=300_000)
    parser.add_argument("--validation-seed", type=int, default=430_000)
    parser.add_argument("--initial-angle-range", type=float, default=0.05)
    parser.add_argument("--force-noise", type=float, default=0.02)
    parser.add_argument("--link-coupling", type=float, default=12.0)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--discrete-action-bins", type=int, default=2)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="parallel")
    parser.add_argument("--mlp-eta", type=float, default=0.08)
    parser.add_argument("--mlp-eta-tick", type=float, default=0.01)
    parser.add_argument("--mlp-sigma", type=float, default=0.08)
    parser.add_argument("--mlp-blend", type=float, default=0.35)
    parser.add_argument("--mlp-hidden-size", type=int, default=16)
    parser.add_argument("--out", default="reports/train_until_solved")
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()
    config = IterationConfig(
        n_poles=args.n_poles,
        target=args.target,
        mode=args.mode,
        action_mode=args.action_mode,
        selection_mode=args.selection_mode,
        budget_episodes=args.budget_episodes,
        train_block_episodes=args.train_block_episodes,
        eval_episodes=args.eval_episodes,
        horizon=args.horizon,
        seed=args.seed,
        validation_seed=args.validation_seed,
        initial_angle_range=args.initial_angle_range,
        force_noise=args.force_noise,
        link_coupling=args.link_coupling,
        force_mag=args.force_mag,
        discrete_action_bins=args.discrete_action_bins,
        dynamics_mode=args.dynamics_mode,
        mlp_eta=args.mlp_eta,
        mlp_eta_tick=args.mlp_eta_tick,
        mlp_sigma=args.mlp_sigma,
        mlp_blend=args.mlp_blend,
        mlp_hidden_size=args.mlp_hidden_size,
        out_dir=args.out,
        resume_checkpoint=args.resume_checkpoint,
    )
    report = run_train_until_solved(config)
    print(json.dumps({key: report[key] for key in ("status", "best_summary", "train_episodes", "wall_clock_seconds")}, indent=2))


if __name__ == "__main__":
    main()
