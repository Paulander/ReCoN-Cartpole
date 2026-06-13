import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_KEYS = [
    "observations",
    "prev_forces",
    "teacher_forces",
    "teacher_actions",
    "returns_to_go",
    "failure_within_k",
    "seeds",
    "sources",
    "rollout_sources",
    "rollout_forces",
    "rollout_actions",
    "motif_scores",
    "episodes",
    "step_indices",
    "sample_weights",
]


REQUIRED_KEYS = [
    "observations",
    "prev_forces",
    "teacher_forces",
    "teacher_actions",
    "returns_to_go",
    "failure_within_k",
    "seeds",
    "sources",
    "rollout_sources",
    "rollout_forces",
    "rollout_actions",
    "episodes",
    "step_indices",
]


@dataclass
class DatasetSpec:
    path: Path
    weight: float = 1.0
    source_prefix: str = ""


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    return data


def row_count(data: dict[str, np.ndarray]) -> int:
    return int(np.asarray(data["teacher_actions"]).shape[0])


def filter_dataset(
    data: dict[str, np.ndarray],
    include_sources: set[str] | None = None,
    exclude_sources: set[str] | None = None,
) -> dict[str, np.ndarray]:
    count = row_count(data)
    mask = np.ones(count, dtype=bool)
    sources = np.asarray(data["sources"]).astype(str)
    if include_sources:
        mask &= np.isin(sources, sorted(include_sources))
    if exclude_sources:
        mask &= ~np.isin(sources, sorted(exclude_sources))
    if not np.any(mask):
        raise ValueError("source filter removed every row")
    filtered: dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        filtered[key] = arr[mask] if arr.shape[:1] == (count,) else arr
    return filtered


def normalize_dataset(data: dict[str, np.ndarray], spec: DatasetSpec, episode_offset: int) -> dict[str, np.ndarray]:
    count = row_count(data)
    result: dict[str, np.ndarray] = {}
    for key in ARRAY_KEYS:
        if key == "motif_scores" and key not in data:
            arr = np.zeros(count, dtype=np.float32)
        elif key == "sample_weights" and key not in data:
            arr = np.ones(count, dtype=np.float32)
        else:
            arr = np.asarray(data[key])
        if arr.shape[:1] != (count,):
            raise ValueError(f"{spec.path} key {key} has incompatible leading shape {arr.shape}")
        if key == "episodes":
            arr = arr.astype(np.int64) + int(episode_offset)
        elif key == "sample_weights":
            arr = np.clip(arr.astype(np.float32) * max(0.0, float(spec.weight)), 0.0, None)
        elif key in ("sources", "rollout_sources") and spec.source_prefix:
            arr = np.asarray([f"{spec.source_prefix}:{value}" for value in arr.astype(str)])
        result[key] = arr
    return result


def next_episode_offset(data: dict[str, np.ndarray], current_offset: int) -> int:
    episodes = np.asarray(data["episodes"], dtype=np.int64)
    if episodes.size == 0:
        return int(current_offset)
    return int(np.max(episodes)) + 1


def merge_datasets(
    specs: list[DatasetSpec],
    include_sources: set[str] | None = None,
    exclude_sources: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not specs:
        raise ValueError("at least one dataset is required")
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
    metadata: dict[str, Any] = {"inputs": [], "include_sources": sorted(include_sources or []), "exclude_sources": sorted(exclude_sources or [])}
    expected_observation_shape: tuple[int, ...] | None = None
    episode_offset = 0
    for spec in specs:
        raw = filter_dataset(load_npz(spec.path), include_sources, exclude_sources)
        obs_shape = tuple(np.asarray(raw["observations"]).shape[1:])
        if expected_observation_shape is None:
            expected_observation_shape = obs_shape
        elif obs_shape != expected_observation_shape:
            raise ValueError(
                f"observation shape mismatch for {spec.path}: {obs_shape} != {expected_observation_shape}"
            )
        normalized = normalize_dataset(raw, spec, episode_offset)
        for key in ARRAY_KEYS:
            parts[key].append(normalized[key])
        count = row_count(normalized)
        metadata["inputs"].append(
            {
                "path": str(spec.path),
                "samples": count,
                "weight": float(spec.weight),
                "source_prefix": spec.source_prefix,
                "episode_offset": int(episode_offset),
                "source_counts": dict(Counter(np.asarray(normalized["sources"]).astype(str).tolist())),
                "sample_weight_mean": float(np.mean(normalized["sample_weights"])) if count else 0.0,
                "sample_weight_max": float(np.max(normalized["sample_weights"])) if count else 0.0,
            }
        )
        episode_offset = next_episode_offset(normalized, episode_offset)
    merged = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    metadata.update(
        {
            "samples": row_count(merged),
            "observation_shape": list(np.asarray(merged["observations"]).shape),
            "source_counts": dict(Counter(np.asarray(merged["sources"]).astype(str).tolist())),
            "sample_weight_mean": float(np.mean(merged["sample_weights"])),
            "sample_weight_max": float(np.max(merged["sample_weights"])),
        }
    )
    return merged, metadata


def parse_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    paths = [Path(item) for item in args.dataset]
    weights = list(args.dataset_weight or [])
    prefixes = list(args.source_prefix or [])
    if weights and len(weights) != len(paths):
        raise ValueError("--dataset-weight must be supplied once per --dataset")
    if prefixes and len(prefixes) != len(paths):
        raise ValueError("--source-prefix must be supplied once per --dataset")
    if not weights:
        weights = [1.0] * len(paths)
    if not prefixes:
        prefixes = [""] * len(paths)
    return [
        DatasetSpec(path=path, weight=weight, source_prefix=prefix)
        for path, weight, prefix in zip(paths, weights, prefixes, strict=True)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_specs(args)
    merged, metadata = merge_datasets(
        specs,
        include_sources=set(args.include_source or []) or None,
        exclude_sources=set(args.exclude_source or []) or None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **merged)
    metadata_path = Path(args.metadata_out) if args.metadata_out else out.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["out"] = str(out)
    metadata["metadata_out"] = str(metadata_path)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--dataset-weight", action="append", type=float, default=[])
    parser.add_argument("--source-prefix", action="append", default=[])
    parser.add_argument("--include-source", action="append", default=[])
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-out", default="")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({"out": result["out"], "samples": result["samples"]}, indent=2))


if __name__ == "__main__":
    main()
