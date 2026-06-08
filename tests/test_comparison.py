from recon_cartpole.training.comparison import run_nway_comparison
from recon_cartpole.training.ppo_baseline import PPOBaselineConfig, ppo_dependency_status, run_ppo_baseline


def test_ppo_baseline_reports_dependency_status_when_unavailable():
    status = ppo_dependency_status()
    result = run_ppo_baseline(PPOBaselineConfig(n_poles=1, horizon=5, train_timesteps=1, eval_seeds=[1]))
    if not status["available"]:
        assert result["status"] == "unavailable"
        assert "uv sync --extra rl" in result["note"]
    else:
        assert result["status"] in {"completed", "unavailable"}


def test_comparison_runner_writes_same_seed_rows(tmp_path):
    result = run_nway_comparison(
        n_values=[1],
        horizon=5,
        eval_episodes=2,
        seed_start=10,
        train_episodes=0,
        ppo_timesteps=1,
        out_dir=tmp_path,
        modes=["baseline_heuristic", "static_recon"],
        include_ppo=True,
    )
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "comparison.md").exists()
    assert len(result["runs"]) == 3
    seeds_by_mode = {row["mode"]: row["seeds"] for row in result["runs"]}
    assert seeds_by_mode["baseline_heuristic"] == seeds_by_mode["static_recon"]
    assert seeds_by_mode["ppo"] == seeds_by_mode["static_recon"]
