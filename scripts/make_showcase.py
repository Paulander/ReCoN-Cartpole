from __future__ import annotations

from pathlib import Path

from recon_cartpole.visualization.dashboard import render_dashboard


def main() -> None:
    Path("reports").mkdir(exist_ok=True)
    render_dashboard({"status": "Run scripts/run_cartpole_v1.py and scripts/run_nlink_demo.py first."}, "reports/showcase.html")
    print("wrote reports/showcase.html")


if __name__ == "__main__":
    main()

