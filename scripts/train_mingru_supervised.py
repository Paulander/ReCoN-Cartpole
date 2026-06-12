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


def estimated_episode_survival(data: dict[str, np.ndarray]) -> np.ndarray:
    count = int(data["teacher_actions"].shape[0])
    if "returns_to_go" not in data:
        return np.zeros(count, dtype=np.float32)
    returns = data["returns_to_go"].astype(np.float32)
    if "step_indices" in data:
        return returns + data["step_indices"].astype(np.float32)
    return returns


def filter_training_data(data: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    count = int(data["teacher_actions"].shape[0])
    min_survival = float(getattr(args, "min_sample_episode_survival", 0.0))
    max_survival = float(getattr(args, "max_sample_episode_survival", 0.0))
    if min_survival <= 0.0 and max_survival <= 0.0:
        return data, {"enabled": False, "input_samples": count, "kept_samples": count}
    survival = estimated_episode_survival(data)
    mask = np.ones(count, dtype=bool)
    if min_survival > 0.0:
        mask &= survival >= min_survival
    if max_survival > 0.0:
        mask &= survival <= max_survival
    if not np.any(mask):
        raise ValueError("sample survival filter removed every row")
    filtered: dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        filtered[key] = arr[mask] if arr.shape[:1] == (count,) else arr
    return filtered, {
        "enabled": True,
        "min_sample_episode_survival": min_survival,
        "max_sample_episode_survival": max_survival,
        "input_samples": count,
        "kept_samples": int(np.sum(mask)),
        "kept_fraction": float(np.mean(mask)),
        "kept_survival_mean": float(np.mean(survival[mask])),
    }


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


def adapt_state_dict_for_input_expansion(model_state: dict[str, Any], checkpoint_state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    adapted = {key: value.clone() for key, value in model_state.items()}
    copied: list[str] = []
    partial: list[dict[str, Any]] = []
    skipped: list[str] = []
    for key, source in checkpoint_state.items():
        if key not in adapted:
            skipped.append(key)
            continue
        target = adapted[key]
        if tuple(target.shape) == tuple(source.shape):
            adapted[key] = source.clone()
            copied.append(key)
        elif target.ndim == source.ndim == 2 and target.shape[0] == source.shape[0]:
            widened = target.clone()
            cols = min(int(target.shape[1]), int(source.shape[1]))
            widened[:, :cols] = source[:, :cols]
            adapted[key] = widened
            partial.append({
                "key": key,
                "source_shape": list(source.shape),
                "target_shape": list(target.shape),
                "copied_columns": cols,
            })
        else:
            skipped.append(key)
    return adapted, {"copied": copied, "partial": partial, "skipped": skipped}


def sample_weights(data: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    count = int(data["teacher_actions"].shape[0])
    weights = np.ones(count, dtype=np.float32)
    dataset_weights: np.ndarray | None = None
    if "sample_weights" in data:
        dataset_weights = np.asarray(data["sample_weights"], dtype=np.float32).reshape(-1)
        if dataset_weights.shape[0] != count:
            raise ValueError("sample_weights length must match teacher_actions")
        dataset_weights = np.clip(dataset_weights, 0.0, None)
    failure_weight = max(0.0, float(getattr(args, "failure_sample_weight", 0.0)))
    late_weight = max(0.0, float(getattr(args, "late_sample_weight", 0.0)))
    low_return_weight = max(0.0, float(getattr(args, "low_return_sample_weight", 0.0)))
    if failure_weight and "failure_within_k" in data:
        weights += failure_weight * np.clip(data["failure_within_k"].astype(np.float32), 0.0, 1.0)
    if late_weight and "step_indices" in data:
        horizon = max(1.0, float(getattr(args, "horizon", 500)))
        progress = np.clip(data["step_indices"].astype(np.float32) / horizon, 0.0, 1.0)
        weights += late_weight * progress
    if low_return_weight and "returns_to_go" in data:
        horizon = max(1.0, float(getattr(args, "horizon", 500)))
        low_return = 1.0 - np.clip(data["returns_to_go"].astype(np.float32) / horizon, 0.0, 1.0)
        weights += low_return_weight * low_return
    if dataset_weights is not None:
        weights *= dataset_weights
    weights /= max(1e-6, float(np.mean(weights)))
    return weights.astype(np.float32)


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install RL extras with `uv sync --extra rl` to train minGRU") from exc

    raw = np.load(args.dataset, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    data, filter_report = filter_training_data(data, args)
    inputs = build_inputs(data, args)
    x, actions, returns, failures = make_sequences(inputs, data, args)
    weights = sample_weights(data, args)
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
    requested_device = str(getattr(args, "device", "auto") or "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model.to(device)
    resume_checkpoint = str(getattr(args, "resume_checkpoint", "") or "")
    resume_report: dict[str, Any] = {"enabled": bool(resume_checkpoint), "partial_input": False}
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if bool(getattr(args, "resume_partial_input", False)):
            adapted, resume_report = adapt_state_dict_for_input_expansion(model.state_dict(), state_dict)
            resume_report.update({"enabled": True, "partial_input": True, "checkpoint": resume_checkpoint})
            model.load_state_dict(adapted)
        else:
            model.load_state_dict(state_dict)
            resume_report.update({"checkpoint": resume_checkpoint})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def batch_loss(batch_idx: np.ndarray, train_mode: bool) -> tuple[torch.Tensor, dict[str, float]]:
        xb = torch.as_tensor(x[batch_idx], dtype=torch.float32, device=device)
        y = torch.as_tensor(actions[batch_idx], dtype=torch.long, device=device)
        rtg = torch.as_tensor(returns[batch_idx], dtype=torch.float32, device=device).unsqueeze(1)
        fail = torch.as_tensor(failures[batch_idx], dtype=torch.float32, device=device).unsqueeze(1)
        weight = torch.as_tensor(weights[batch_idx], dtype=torch.float32, device=device)
        hidden = torch.zeros(xb.shape[0], args.hidden_size, dtype=torch.float32, device=device)
        logits, value, failure, confidence, _hidden = model(xb, hidden)
        ce_each = F.cross_entropy(logits, y, reduction="none")
        ce = torch.mean(ce_each * weight)
        value_loss = torch.mean(F.mse_loss(value, rtg, reduction="none") * weight.unsqueeze(1))
        failure_loss = torch.mean(F.binary_cross_entropy(failure, fail, reduction="none") * weight.unsqueeze(1))
        confidence_target = 1.0 - fail
        confidence_loss = torch.mean(F.binary_cross_entropy(confidence, confidence_target, reduction="none") * weight.unsqueeze(1))
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
        "resume_checkpoint": resume_checkpoint,
        "resume_load": resume_report,
        "samples": int(x.shape[0]),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "config": config.__dict__,
        "history": history,
        "device": str(device),
        "sample_filter": filter_report,
        "sample_weighting": {
            "failure_sample_weight": max(0.0, float(getattr(args, "failure_sample_weight", 0.0))),
            "late_sample_weight": max(0.0, float(getattr(args, "late_sample_weight", 0.0))),
            "low_return_sample_weight": max(0.0, float(getattr(args, "low_return_sample_weight", 0.0))),
            "dataset_sample_weights": "sample_weights" in data,
            "dataset_sample_weight_mean": float(np.mean(data["sample_weights"])) if "sample_weights" in data else 1.0,
            "dataset_sample_weight_max": float(np.max(data["sample_weights"])) if "sample_weights" in data else 1.0,
            "mean_weight": float(np.mean(weights)),
            "max_weight": float(np.max(weights)) if weights.size else 0.0,
        },
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
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-partial-input", action="store_true", default=False)
    parser.add_argument("--out", default="reports/mingru_supervised")
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--force-mag", type=float, default=10.0)
    parser.add_argument("--discrete-action-bins", type=int, default=5)
    parser.add_argument("--observation-mode", choices=["env", "normalized_raw", "normalized_raw_prev_force", "normalized_raw4", "normalized_raw4_prev_force", "normalized_raw4_subchains", "normalized_raw4_subchains_prev_force"], default="normalized_raw")
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
    parser.add_argument("--min-sample-episode-survival", type=float, default=0.0)
    parser.add_argument("--max-sample-episode-survival", type=float, default=0.0)
    parser.add_argument("--failure-sample-weight", type=float, default=0.0)
    parser.add_argument("--late-sample-weight", type=float, default=0.0)
    parser.add_argument("--low-return-sample-weight", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9117)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps({"checkpoint_path": report["checkpoint_path"], "samples": report["samples"]}, indent=2))


if __name__ == "__main__":
    main()
