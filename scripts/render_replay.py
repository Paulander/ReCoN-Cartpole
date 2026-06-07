from __future__ import annotations

import argparse

from recon_cartpole.recon.trace_db import load_trace
from recon_cartpole.visualization.physics_render import render_trace_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render_trace_html(load_trace(args.trace), args.out)


if __name__ == "__main__":
    main()

