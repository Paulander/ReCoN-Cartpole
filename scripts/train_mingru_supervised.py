from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recon_cartpole.recon.mingru_terminal import MinGRUTerminal, MinGRUTerminalConfig


def observation_mode_has_prev_force(mode: str) -> bool:
    return "prev_force" in str(mode)


def build_inputs(data: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    parts = [data["observations"].astype(np.float32)]
    if args.include_prev_force and not observation_mode_has_prev_force(args.observation_mode):
        parts.append((data["prev_forces"].astype(np.float32) / max(args.force_mag, 1e-9))[:, None])
    if args.include_context:
        # Supervised traces do not yet store ReCoN risk/regime context; keep the input shape
        # checkpoint-compatible and let future dataset builders fill these columns.
        parts.append(np.zeros((data["observations"].shape[0], 3), dtype=np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def sequence_indices(episodes: np.ndarray, seq_len: int) -> list[list[int]]:
    seq_len = max(1, int(seq_len))
    result: list[list[int]] = []
    for idx in range(len(episodes)):
        episode = episodes[idx]
        window = [idx]
        cursor = idx - 1
        while cursor >= 0 and episodes[cursor] == episode and len(window) < seq_len:
            window.append(cursor)
            cursor -= 1
        window = list(reversed(window))
        if len(window) < seq_len:
            window = [window[0]] * (seq_len - len(window)) + window
        result.append(window)
    return result


def make_sequences(inputs: np.ndarray, data: dict[str, np.ndarray], args: argparse.Namespace):
    indices = sequence_indices(data["episodes"], args.sequence_length)
    x = np.stack([inputs[idxs] for idxs in indices]).astype(np.float32)
    actions = data["teacher_actions"].astype(np.int64)
    if args.discrete_action_bins > 0:
        actions = np.clip(actions, 0, args.discrete_action_bins - 1)
    returns = data["returns_to_go"].astype(np.float32)
    returns = returns / max(1.0, float(args.horizon))
    failures = data["failure_within_k"].astype(np.float32)
    return x, actions, returns, failures


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install RL extras with `uv sync --extra rl` to train minGRU") from exc

    raw = np.load(args.dataset, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    inputs = build_inputs(data, args)
    x, actions, returns, failures = make_sequences(inputs, data, args)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(x.shape[0])
    split = int(x.shape[0] * (1.0 - args.validation_fraction))
    train_idx = order[:split]
    val_idx = order[split:] if split < len(order) else order[:0]

    config = MinGRUTerminalConfig(
        enabled=True,
        hidden_size=args.hidden_size,
        sequence_length=args.sequence_length,
        observation_mode=args.observation_mode,
        include_prev_force=args.include_prev_force,
        include_context=args.include_context,
        blend=args.blend,
        scope=args.scope,
        confidence_floor=args.confidence_floor,
    )
    terminal = MinGRUTerminal(args.n_poles, args.force_mag, args.discrete_action_bins, config)
    model = terminal.model
    assert model is not None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def batch_loss(batch_idx: np.ndarray, train_mode: bool) -> tuple[torch.Tensor, dict[str, float]]:
        xb = torch.as_tensor(x[batch_idx], dtype=torch.float32)
        y = torch.as_tensor(actions[batch_idx], dtype=torch.long)
        rtg = torch.as_tensor(returns[batch_idx], dtype=torch.float32).unsqueeze(1)
        fail = torch.as_tensor(failures[batch_idx], dtype=torch.float32).unsqueeze(1)
        hidden = torch.zeros(xb.shape[0], args.hidden_size, dtype=torch.float32)
        logits, value, failure, confidence, _hidden = model(xb, hidden)
        ce = F.cross_entropy(logits, y)
        value_loss = F.mse_loss(value, rtg)
        failure_loss = F.binary_cross_entropy(failure, fail)
        confidence_target = 1.0 - fail
        confidence_loss = F.binary_cross_entropy(confidence, confidence_target)
        loss = ce + args.value_weight * value_loss + args.failure_weight * failure_loss + args.confidence_weight * confidence_loss
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        pred = torch.argmax(logits, dim=1)
        metrics = {
            "loss": float(loss.detach()),
            "action_accuracy": float((pred == y).float().mean().detach()),
            "ce": float(ce.detach()),
            "value_loss": float(value_loss.detach()),
            "failure_loss": float(failure_loss.detach()),
        }
        return loss, metrics

    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        batch_metrics: list[dict[str, float]] = []
        for start in range(0, len(train_idx), args.batch_size):
            batch = train_idx[start : start + args.batch_size]
            if len(batch) == 0:
                continue
            _loss, metrics = batch_loss(batch, True)
            batch_metrics.append(metrics)
        row = {f"train_{k}": float(np.mean([m[k] for m in batch_metrics])) for k in batch_metrics[0]} if batch_metrics else {}
        if len(val_idx):
            with torch.no_grad():
                _loss, val_metrics = batch_loss(val_idx, False)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["epoch"] = float(epoch + 1)
        history.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "mingru_terminal.pt"
    terminal.save_checkpoint(str(checkpoint_path))
    report = {
        "checkpoint_path": str(checkpoint_path),
        "dataset": args.dataset,
        "samples": int(x.shape[0]),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "config": config.__dict__,
        "history": history,
        "mechanisms": {
            "minGRU_terminal": True,
            "supervised_imitation": True,
            "edge_plasticity": False,
            "bandit_persistence": False,
            "slow_consolidation": False,
            "gain_mutation": False,
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="reports/mingru_supervised")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force"], default="normalized_raw")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--include-prev-force", action="store_true", default=True)
    parser.add_argument("--no-prev-force", dest="include_prev_force", action="store_false")
    parser.add_argument("--include-context", action="store_true", default=True)
    parser.add_argument("--no-context", dest="include_context", action="store_false")
    parser.add_argument("--scope", choices=["stabilize_chain", "selected", "all"], default="stabilize_chain")
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--confidence-floor", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--value-weight", type=float, default=0.05)
    parser.add_argument("--failure-weight", type=float, default=0.10)
    parser.add_argument("--confidence-weight", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=9117)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps({"checkpoint_path": report["checkpoint_path"], "samples": report["samples"]}, indent=2))


if __name__ == "__main__":
    main()
