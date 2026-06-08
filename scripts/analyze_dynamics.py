from __future__ import annotations

import argparse
import json

import numpy as np

from recon_cartpole.envs.cartpole_n import CartPoleNConfig, CartPoleNEnv


def linearize(config: CartPoleNConfig, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    dim = 2 + 2 * config.n_poles

    def step_from(state: np.ndarray, force: float) -> np.ndarray:
        env = CartPoleNEnv(config)
        env.state = np.asarray(state, dtype=float).copy()
        env._integrate(float(force))
        return env.state.copy()

    origin = np.zeros(dim, dtype=float)
    a_matrix = np.zeros((dim, dim), dtype=float)
    b_matrix = np.zeros((dim, 1), dtype=float)
    for idx in range(dim):
        delta = np.zeros(dim, dtype=float)
        delta[idx] = eps
        a_matrix[:, idx] = (step_from(origin + delta, 0.0) - step_from(origin - delta, 0.0)) / (2.0 * eps)
    b_matrix[:, 0] = (step_from(origin, eps) - step_from(origin, -eps)) / (2.0 * eps)
    return a_matrix, b_matrix


def controllability_summary(config: CartPoleNConfig) -> dict[str, object]:
    a_matrix, b_matrix = linearize(config)
    dim = a_matrix.shape[0]
    controllability = np.concatenate([np.linalg.matrix_power(a_matrix, power) @ b_matrix for power in range(dim)], axis=1)
    singular_values = np.linalg.svd(controllability, compute_uv=False)
    eigvals = np.linalg.eigvals(a_matrix)
    return {
        "n_poles": config.n_poles,
        "dynamics_mode": config.dynamics_mode,
        "state_dim": dim,
        "rank_tol_1e_8": int(np.linalg.matrix_rank(controllability, tol=1e-8)),
        "rank_tol_1e_10": int(np.linalg.matrix_rank(controllability, tol=1e-10)),
        "singular_values": [float(value) for value in singular_values],
        "max_abs_eigenvalue": float(np.max(np.abs(eigvals))),
        "input_vector": [float(value) for value in b_matrix[:, 0]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-poles", type=int, default=4)
    parser.add_argument("--dynamics-mode", choices=["parallel", "serial_lagrange"], default="parallel")
    parser.add_argument("--link-coupling", type=float, default=12.0)
    args = parser.parse_args()
    config = CartPoleNConfig(
        n_poles=args.n_poles,
        dynamics_mode=args.dynamics_mode,
        link_coupling=args.link_coupling,
        action_mode="continuous",
    )
    print(json.dumps(controllability_summary(config), indent=2))


if __name__ == "__main__":
    main()
