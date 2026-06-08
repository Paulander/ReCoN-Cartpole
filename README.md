# ReCoN CartPole

This project reuses the clean `Paulander/recon-lite` library as a vendored,
domain-neutral base and adds a CartPole/N-link control showcase around it.

The first milestone is intentionally honest:

- `GymCartPoleAdapter` wraps `gymnasium.make("CartPole-v1")`.
- `CartPoleNEnv` provides a Gymnasium-compatible custom environment for 1..N poles.
- `ReConCartPoleController` runs a generated ReCoN graph, a heuristic baseline,
  UCB sibling selection, fast edge plasticity, and goal-aware modulation.
- Replay traces can be exported as JSON and rendered as standalone HTML.
- `recon_slow` enables slow consolidation; other modes keep it disabled.

## Quick Start

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_cartpole_v1.py --episodes 3 --render-html reports/cartpole_v1.html
uv run python scripts/run_nlink_demo.py --n-poles 2 --episodes 1 --render-html reports/nlink_2.html
```

The controller modes are:

- `baseline_random`
- `baseline_heuristic`
- `static_recon`
- `recon_fast`
- `recon_bandit`
- `recon_fast_bandit`
- `recon_slow`
- `gain_search_only`
- `gain_search_recon_fast_bandit`

Reports list active mechanisms separately: edge plasticity, bandit persistence, slow consolidation, and external gain mutation. Do not describe a gain-search-only improvement as "ReCoN learned" unless one of the ReCoN learning mechanisms was active on the held-out run.

The custom N-link dynamics are a stable control benchmark scaffold, not yet a
validated multibody physics paper. The docs call this out because the research
claim should come from held-out metrics, not a cherry-picked animation.

