from __future__ import annotations

from .evaluate import evaluate


ABLATION_MODES = [
    "baseline_random",
    "baseline_heuristic",
    "static_recon",
    "recon_fast",
    "recon_bandit",
    "recon_fast_bandit",
    "recon_slow",
]


def run_ablations(env_name: str = "cartpole_n", n_poles: int = 1, episodes: int = 20):
    return [evaluate(env_name, mode, n_poles=n_poles, episodes=episodes) for mode in ABLATION_MODES]

