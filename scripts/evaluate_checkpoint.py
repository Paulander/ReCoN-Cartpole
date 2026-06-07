from __future__ import annotations

import argparse
import json

from recon_cartpole.training.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="cartpole_n")
    parser.add_argument("--mode", default="static_recon")
    parser.add_argument("--n-poles", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.env, args.mode, args.n_poles, args.episodes), indent=2))


if __name__ == "__main__":
    main()

