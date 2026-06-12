from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.control.actuators import action_from_force
from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv
from recon_cartpole.recon.mingru_terminal import MinGRUTerminal, MinGRUTerminalConfig

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_recurrent_terminal_ladder import evaluate_pure_mingru, evaluate_recon_mingru  # noqa: E402


def make_env(args: argparse.Namespace) -> CartPoleNEnv:
    return CartPoleNEnv(
        CartPoleNConfig(
            n_poles=args.n_poles,
            horizon=args.horizon,
            dt=args.dt,
            dynamics_mode=args.dynamics_mode,
            action_mode="discrete",
            discrete_action_bins=args.discrete_action_bins,
            force_mag=args.force_mag,
            initial_angle_range=args.initial_angle_range,
            force_noise=args.force_noise,
            link_coupling=args.link_coupling,
        )
    )


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + float(gamma) * running
        out[idx] = running
    return out


def seed_values(args: argparse.Namespace) -> list[int]:
    seed_list = str(getattr(args, "seed_list", "") or "").strip()
    if seed_list:
        raw = Path(seed_list).read_text(encoding="utf-8")
        seeds: list[int] = []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("hard_seeds", "seeds", "tail_seeds"):
                if isinstance(payload.get(key), list):
                    seeds = [int(item) for item in payload[key]]
                    break
        elif isinstance(payload, list):
            seeds = [int(item) for item in payload]
        if not seeds:
            for item in raw.replace(",", "\n").splitlines():
                value = item.strip()
                if value:
                    seeds.append(int(value))
        return seeds[: int(args.train_episodes)]
    return [int(args.seed_start) + idx for idx in range(int(args.train_episodes))]


def terminal_config(args: argparse.Namespace, checkpoint_path: str = "") -> MinGRUTerminalConfig:
    return MinGRUTerminalConfig(
        enabled=True,
        hidden_size=args.hidden_size,
        sequence_length=args.sequence_length,
        observation_mode=args.observation_mode,
        include_prev_force=args.include_prev_force,
        include_context=args.include_context,
        include_motif_score=bool(getattr(args, "include_motif_score", False)),
        motif_model_path=str(getattr(args, "motif_model_path", "") or ""),
        motif_score_scale=float(getattr(args, "motif_score_scale", 10.0)),
        blend=args.blend,
        scope=args.scope,
        confidence_floor=args.confidence_floor,
        passthrough_enabled=bool(getattr(args, "passthrough_enabled", False)),
        passthrough_confidence_floor=float(getattr(args, "passthrough_confidence_floor", 0.05)),
        passthrough_logit_margin_floor=float(getattr(args, "passthrough_logit_margin_floor", 0.0)),
        checkpoint_path=checkpoint_path,
    )


def make_terminal(args: argparse.Namespace, checkpoint_path: str) -> MinGRUTerminal:
    return MinGRUTerminal(
        args.n_poles,
        args.force_mag,
        args.discrete_action_bins,
        terminal_config(args, checkpoint_path),
    )


def window_tensor(history: list[np.ndarray], seq_len: int, device: Any):
    import torch

    seq_len = max(1, int(seq_len))
    frames = history[-seq_len:]
    if len(frames) < seq_len:
        frames = [frames[0]] * (seq_len - len(frames)) + frames
    return torch.as_tensor(np.stack(frames), dtype=torch.float32, device=device).unsqueeze(0)


def collect_episode(
    args: argparse.Namespace,
    terminal: MinGRUTerminal,
    ref_terminal: MinGRUTerminal,
    device: Any,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    assert terminal.model is not None
    assert ref_terminal.model is not None
    env = make_env(args)
    obs, info = env.reset(seed=seed)
    terminal.reset()
    ref_terminal.reset()
    history: list[np.ndarray] = []
    ref_history: list[np.ndarray] = []
    log_probs = []
    values = []
    entropies = []
    kls = []
    rewards: list[float] = []
    actions: list[int] = []
    forces = np.linspace(-args.force_mag, args.force_mag, args.discrete_action_bins)
    total_env_reward = 0.0
    success = False

    for step in range(int(args.horizon)):
        raw = info.get("raw_state")
        vec = terminal.observation_vector(obs, raw, {})
        ref_vec = ref_terminal.observation_vector(obs, raw, {})
        history.append(vec)
        ref_history.append(ref_vec)
        x = window_tensor(history, args.sequence_length, device)
        ref_x = window_tensor(ref_history, args.sequence_length, device)
        hidden = torch.zeros(1, int(args.hidden_size), dtype=torch.float32, device=device)
        logits, value, _failure, _confidence, _ = terminal.model(x, hidden)
        with torch.no_grad():
            ref_logits, _rv, _rf, _rc, _rh = ref_terminal.model(ref_x, hidden)
        dist = torch.distributions.Categorical(logits=logits)
        if bool(args.sample_actions):
            action_t = dist.sample()
        else:
            action_t = torch.argmax(logits, dim=1)
        ref_dist = torch.distributions.Categorical(logits=ref_logits)
        action = int(action_t.detach().cpu().reshape(-1)[0])
        force = float(forces[max(0, min(len(forces) - 1, action))])
        env_action = action_from_force(force, "discrete", args.force_mag, args.discrete_action_bins)
        obs, env_reward, terminated, truncated, info = env.step(env_action)
        total_env_reward += float(env_reward)
        terminal.prev_force = force
        ref_terminal.prev_force = force
        shaped = float(env_reward)
        log_probs.append(dist.log_prob(action_t).reshape(()))
        values.append(value.reshape(()))
        entropies.append(dist.entropy().reshape(()))
        kls.append(torch.distributions.kl_divergence(dist, ref_dist).reshape(()))
        rewards.append(shaped)
        actions.append(action)
        if terminated or truncated:
            success = bool(step + 1 >= int(args.horizon) and total_env_reward >= float(args.horizon) - 1.0)
            if success:
                rewards[-1] += float(args.success_bonus)
            else:
                rewards[-1] -= float(args.failure_penalty)
            break
    else:
        success = True
        if rewards:
            rewards[-1] += float(args.success_bonus)

    returns = torch.as_tensor(
        discounted_returns(rewards, float(args.gamma)) / max(1.0, float(args.horizon)),
        dtype=torch.float32,
        device=device,
    )
    value_t = torch.stack(values)
    log_prob_t = torch.stack(log_probs)
    entropy_t = torch.stack(entropies)
    kl_t = torch.stack(kls)
    advantage = returns - value_t.detach()
    if bool(args.normalize_advantage) and advantage.numel() > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
    policy_loss = -(log_prob_t * advantage).mean()
    value_loss = F.mse_loss(value_t, returns)
    entropy = entropy_t.mean()
    kl = kl_t.mean()
    loss = policy_loss + float(args.value_coef) * value_loss + float(args.kl_coef) * kl - float(args.ent_coef) * entropy
    return {
        "loss": loss,
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "steps": len(rewards),
        "return": float(total_env_reward),
        "success": success,
        "seed": int(seed),
        "actions": actions,
    }


def eval_seeds(args: argparse.Namespace) -> list[int]:
    seeds: list[int] = []
    for start in args.final_seed_starts:
        seeds.extend(int(start) + idx for idx in range(int(args.final_eval_episodes)))
    return seeds


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install RL extras with torch to fine-tune minGRU on-policy") from exc

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
    terminal.model.train()
    ref_terminal.model.eval()
    for param in ref_terminal.model.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(terminal.model.parameters(), lr=float(args.learning_rate))

    seeds = seed_values(args)
    history: list[dict[str, Any]] = []
    batch_losses = []
    batch_rows = []
    episodes_seen = 0
    for index, seed in enumerate(seeds, start=1):
        row = collect_episode(args, terminal, ref_terminal, device, seed)
        batch_losses.append(row["loss"])
        batch_rows.append(row)
        if len(batch_losses) >= int(args.update_episodes) or index == len(seeds):
            loss = torch.stack(batch_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(terminal.model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            episodes_seen += len(batch_rows)
            metric = {
                "episodes": episodes_seen,
                "batch_size": len(batch_rows),
                "loss": float(loss.detach().cpu()),
                "mean_steps": float(np.mean([r["steps"] for r in batch_rows])),
                "success_rate": float(np.mean([r["success"] for r in batch_rows])),
                "policy_loss": float(np.mean([r["policy_loss"] for r in batch_rows])),
                "value_loss": float(np.mean([r["value_loss"] for r in batch_rows])),
                "entropy": float(np.mean([r["entropy"] for r in batch_rows])),
                "kl": float(np.mean([r["kl"] for r in batch_rows])),
            }
            history.append(metric)
            if bool(getattr(args, "progress", True)):
                print(json.dumps({"episodes": episodes_seen, "mean_steps": metric["mean_steps"], "success_rate": metric["success_rate"], "loss": metric["loss"]}), flush=True)
            batch_losses = []
            batch_rows = []

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "mingru_onpolicy.pt"
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
        "train_seed_count": len(seeds),
        "train_seed_start": int(args.seed_start),
        "seed_list": str(getattr(args, "seed_list", "") or ""),
        "history": history,
        "pure_mingru_policy": pure,
        "recon_mingru_terminal": recon,
        "eval_seeds": seeds_eval,
        "config": vars(args),
        "mechanisms": {
            "minGRU_terminal": True,
            "on_policy_actor_critic": True,
            "reference_kl_preservation": float(args.kl_coef) > 0.0,
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
    parser = argparse.ArgumentParser(description="Conservative on-policy actor-critic fine-tuning for minGRU terminals.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--out", default="reports/mingru_onpolicy")
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
    parser.add_argument("--seed-start", type=int, default=6_000_000)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--update-episodes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--failure-penalty", type=float, default=25.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--value-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.002)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--sample-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-seed", type=int, default=9301)
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
