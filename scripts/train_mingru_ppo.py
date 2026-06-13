from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from recon_cartpole.control.actuators import action_from_force

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_mingru_onpolicy import (  # noqa: E402
    discounted_returns,
    eval_seeds,
    make_env,
    make_terminal,
    seed_values,
    window_tensor,
)
from train_recurrent_terminal_ladder import evaluate_pure_mingru, evaluate_recon_mingru  # noqa: E402


class RolloutBatch(NamedTuple):
    sequences: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    seeds: np.ndarray
    steps: np.ndarray
    episode_steps: list[int]
    successes: list[bool]


def ppo_clipped_policy_loss(ratio: Any, advantages: Any, clip_range: float):
    import torch

    clipped = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
    return -torch.mean(torch.minimum(ratio * advantages, clipped * advantages))


def normalize(values: np.ndarray, enabled: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not enabled or values.size <= 1:
        return values
    return ((values - float(np.mean(values))) / (float(np.std(values)) + 1e-6)).astype(np.float32)


def collect_rollouts(args: argparse.Namespace, terminal: Any, device: Any, seeds: list[int]) -> RolloutBatch:
    import torch

    assert terminal.model is not None
    terminal.model.eval()
    force_values = np.linspace(-args.force_mag, args.force_mag, args.discrete_action_bins)
    sequences: list[np.ndarray] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    old_values: list[float] = []
    rewards_by_episode: list[list[float]] = []
    seed_rows: list[int] = []
    step_rows: list[int] = []
    episode_steps: list[int] = []
    successes: list[bool] = []

    for seed in seeds:
        env = make_env(args)
        obs, info = env.reset(seed=int(seed))
        terminal.reset()
        history: list[np.ndarray] = []
        ep_indices: list[int] = []
        ep_rewards: list[float] = []
        total_env_reward = 0.0
        success = False
        for step in range(int(args.horizon)):
            raw = info.get("raw_state")
            vector = terminal.observation_vector(obs, raw, {})
            history.append(vector)
            x = window_tensor(history, int(args.sequence_length), device)
            hidden = torch.zeros(1, int(args.hidden_size), dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value, _failure, _confidence, _hidden = terminal.model(x, hidden)
                dist = torch.distributions.Categorical(logits=logits)
                action_t = dist.sample() if bool(args.sample_actions) else torch.argmax(logits, dim=1)
                log_prob = dist.log_prob(action_t).reshape(()).detach().cpu().item()
                value_f = value.reshape(()).detach().cpu().item()
            action = int(action_t.detach().cpu().reshape(-1)[0])
            force = float(force_values[max(0, min(len(force_values) - 1, action))])
            env_action = action_from_force(force, "discrete", args.force_mag, args.discrete_action_bins)
            next_obs, env_reward, terminated, truncated, info = env.step(env_action)
            total_env_reward += float(env_reward)
            shaped = float(env_reward)
            if float(args.late_survival_bonus) and step + 1 >= int(args.horizon) * float(args.late_survival_start_fraction):
                shaped += float(args.late_survival_bonus)

            sequences.append(x.detach().cpu().numpy().reshape(int(args.sequence_length), -1).astype(np.float32))
            actions.append(action)
            old_log_probs.append(float(log_prob))
            old_values.append(float(value_f))
            seed_rows.append(int(seed))
            step_rows.append(int(step))
            ep_indices.append(len(sequences) - 1)
            ep_rewards.append(float(shaped))

            terminal.prev_force = force
            obs = next_obs
            if terminated or truncated:
                success = bool(step + 1 >= int(args.horizon) and total_env_reward >= float(args.horizon) - 1.0)
                if ep_rewards:
                    ep_rewards[-1] += float(args.success_bonus) if success else -float(args.failure_penalty)
                break
        else:
            success = True
            if ep_rewards:
                ep_rewards[-1] += float(args.success_bonus)

        episode_steps.append(len(ep_rewards))
        successes.append(bool(success))
        rewards_by_episode.append(ep_rewards)

    returns = np.zeros(len(sequences), dtype=np.float32)
    cursor = 0
    for ep_rewards in rewards_by_episode:
        count = len(ep_rewards)
        if count:
            returns[cursor : cursor + count] = discounted_returns(ep_rewards, float(args.gamma)) / max(1.0, float(args.horizon))
        cursor += count
    old_values_arr = np.asarray(old_values, dtype=np.float32)
    advantages = normalize(returns - old_values_arr, bool(args.normalize_advantage))
    return RolloutBatch(
        sequences=np.asarray(sequences, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        old_log_probs=np.asarray(old_log_probs, dtype=np.float32),
        old_values=old_values_arr,
        returns=returns,
        advantages=advantages,
        seeds=np.asarray(seed_rows, dtype=np.int64),
        steps=np.asarray(step_rows, dtype=np.int64),
        episode_steps=episode_steps,
        successes=successes,
    )


def ppo_update(args: argparse.Namespace, model: Any, ref_model: Any, batch: RolloutBatch, device: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.train()
    if ref_model is not None:
        ref_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    x = torch.as_tensor(batch.sequences, dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(batch.old_log_probs, dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=device)
    indices = np.arange(batch.actions.shape[0])
    rng = np.random.default_rng(int(args.train_seed))
    metrics: list[dict[str, float]] = []
    for _epoch in range(int(args.ppo_epochs)):
        rng.shuffle(indices)
        for start in range(0, len(indices), int(args.minibatch_size)):
            batch_idx = indices[start : start + int(args.minibatch_size)]
            if batch_idx.size == 0:
                continue
            xb = x[batch_idx]
            yb = actions[batch_idx]
            hidden = torch.zeros(xb.shape[0], int(args.hidden_size), dtype=torch.float32, device=device)
            logits, value, _failure, _confidence, _hidden = model(xb, hidden)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(yb)
            ratio = torch.exp(log_probs - old_log_probs[batch_idx])
            policy_loss = ppo_clipped_policy_loss(ratio, advantages[batch_idx], float(args.clip_range))
            value_loss = F.mse_loss(value.reshape(-1), returns[batch_idx])
            entropy = dist.entropy().mean()
            ref_kl = torch.zeros((), dtype=torch.float32, device=device)
            if ref_model is not None and float(args.ref_kl_coef) > 0.0:
                with torch.no_grad():
                    ref_logits, _rv, _rf, _rc, _rh = ref_model(xb, hidden)
                    ref_dist = torch.distributions.Categorical(logits=ref_logits)
                ref_kl = torch.distributions.kl_divergence(dist, ref_dist).mean()
            loss = policy_loss + float(args.value_coef) * value_loss - float(args.ent_coef) * entropy + float(args.ref_kl_coef) * ref_kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            approx_kl = torch.mean(old_log_probs[batch_idx] - log_probs).detach()
            metrics.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "ref_kl": float(ref_kl.detach().cpu()),
                    "approx_kl": float(approx_kl.cpu()),
                }
            )
            if float(args.target_kl) > 0.0 and float(approx_kl.cpu()) > float(args.target_kl):
                break
    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]} if metrics else {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install RL extras with torch to fine-tune minGRU with PPO") from exc

    requested_device = str(getattr(args, "device", "auto") or "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    torch.manual_seed(int(args.train_seed))
    np.random.seed(int(args.train_seed))

    terminal = make_terminal(args, args.checkpoint_path)
    ref_terminal = make_terminal(args, args.checkpoint_path)
    assert terminal.model is not None and ref_terminal.model is not None
    terminal.model.to(device)
    ref_terminal.model.to(device)
    for param in ref_terminal.model.parameters():
        param.requires_grad_(False)

    all_seeds = seed_values(args)
    history: list[dict[str, Any]] = []
    for iteration in range(int(args.iterations)):
        start = iteration * int(args.rollout_episodes)
        rollout_seeds = all_seeds[start : start + int(args.rollout_episodes)]
        if not rollout_seeds:
            break
        batch = collect_rollouts(args, terminal, device, rollout_seeds)
        metrics = ppo_update(args, terminal.model, ref_terminal.model, batch, device)
        row = {
            "iteration": iteration + 1,
            "episodes": len(rollout_seeds),
            "mean_steps": float(np.mean(batch.episode_steps)) if batch.episode_steps else 0.0,
            "success_rate": float(np.mean(batch.successes)) if batch.successes else 0.0,
            "sample_count": int(batch.actions.shape[0]),
            **metrics,
        }
        history.append(row)
        if bool(args.progress):
            print(json.dumps(row), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "mingru_ppo.pt"
    terminal.save_checkpoint(str(checkpoint_path))

    ladder_args = argparse.Namespace(
        n_poles=args.n_poles,
        horizon=args.horizon,
        dt=args.dt,
        dynamics_mode=args.dynamics_mode,
        discrete_action_bins=args.discrete_action_bins,
        force_mag=args.force_mag,
        initial_angle_range=args.initial_angle_range,
        force_noise=args.force_noise,
        link_coupling=args.link_coupling,
        observation_mode=args.observation_mode,
        include_prev_force=args.include_prev_force,
        include_context=args.include_context,
        include_motif_score=args.include_motif_score,
        motif_model_path=args.motif_model_path,
        motif_score_scale=args.motif_score_scale,
        blend=args.blend,
        scope=args.scope,
        confidence_floor=args.confidence_floor,
        passthrough_enabled=args.passthrough_enabled,
        passthrough_confidence_floor=args.passthrough_confidence_floor,
        passthrough_logit_margin_floor=args.passthrough_logit_margin_floor,
        selection_mode=args.selection_mode,
    )
    seeds_eval = eval_seeds(args)
    pure = evaluate_pure_mingru(str(checkpoint_path), ladder_args, seeds_eval, args.hidden_size, args.sequence_length)
    recon = evaluate_recon_mingru(str(checkpoint_path), ladder_args, seeds_eval, args.hidden_size, args.sequence_length)
    report = {
        "status": "completed",
        "checkpoint_path": str(checkpoint_path),
        "start_checkpoint_path": str(args.checkpoint_path),
        "train_seed_count": len(all_seeds),
        "seed_list": str(getattr(args, "seed_list", "") or ""),
        "history": history,
        "pure_mingru_policy": pure,
        "recon_mingru_terminal": recon,
        "eval_seeds": seeds_eval,
        "config": vars(args),
        "mechanisms": {
            "minGRU_terminal": True,
            "ppo_clipped_policy_gradient": True,
            "reference_kl_preservation": float(args.ref_kl_coef) > 0.0,
            "edge_plasticity": False,
            "bandit_persistence": False,
            "slow_consolidation": False,
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO-style clipped fine-tuning for minGRU recurrent terminals.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--out", default="reports/mingru_ppo")
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
    parser.add_argument("--observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force", "normalized_raw4_subchains", "normalized_raw4_subchains_prev_force"], default="normalized_raw4_subchains_prev_force")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--include-prev-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-motif-score", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--motif-model-path", default="")
    parser.add_argument("--motif-score-scale", type=float, default=10.0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--passthrough-confidence-floor", type=float, default=0.05)
    parser.add_argument("--passthrough-logit-margin-floor", type=float, default=0.0)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=7_000_000)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--rollout-episodes", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--clip-range", type=float, default=0.05)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--failure-penalty", type=float, default=25.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--late-survival-bonus", type=float, default=0.0)
    parser.add_argument("--late-survival-start-fraction", type=float, default=0.80)
    parser.add_argument("--value-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.002)
    parser.add_argument("--ref-kl-coef", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--sample-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-seed", type=int, default=9401)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-seed-starts", type=int, nargs="+", default=[1_900_000, 2_000_000, 2_100_000, 2_200_000])
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "out": report["config"]["out"],
                "checkpoint_path": report["checkpoint_path"],
                "success": report["recon_mingru_terminal"].get("success_rate", 0.0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
