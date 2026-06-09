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
- `recon_mlp_terminal`
- `recon_policy_terminal`

Reports list active mechanisms separately: edge plasticity, bandit persistence, slow consolidation, and external gain mutation. Do not describe a gain-search-only improvement as "ReCoN learned" unless one of the ReCoN learning mechanisms was active on the held-out run.

## ReCoN vs PPO Comparison

Use `run_comparison.py` to produce same-seed N=2/3/4 comparison tables across ReCoN modes and an optional PPO baseline:

```bash
uv run python scripts/run_comparison.py --n-values 2 3 4 --horizon 500 --eval-episodes 300 --train-episodes 1000 --ppo-timesteps 200000 --ppo-device cpu --out reports/recon_vs_ppo_full
```

PPO uses optional dependencies. If `torch`/`stable-baselines3` are missing, the report keeps an explicit `ppo` row with `status: unavailable`. Install with `uv sync --extra rl` before making PPO performance claims.

## Policy Terminal Training

`recon_policy_terminal` lets a saved PPO policy emit the `stabilize_chain` terminal proposal while ReCoN still handles graph routing and arbitration. This is a learned neural terminal inside the ReCoN scaffold, not pure symbolic ReCoN.

```bash
uv sync --extra rl
uv run python scripts/train_policy_terminal.py --n-poles 4 --dynamics-mode serial_lagrange --dt 0.0005 --action-mode discrete --discrete-action-bins 5 --reward-mode upright_shaping --timesteps 50000 --net-arch 64,64 --gamma 0.995 --n-steps 512 --batch-size 256 --out reports/policy_terminal_n4
uv run python scripts/train_policy_terminal.py --model-path reports/policy_terminal_n4/ppo_policy_terminal.zip --n-poles 4 --dynamics-mode serial_lagrange --dt 0.0005 --action-mode discrete --discrete-action-bins 5 --eval-episodes 300 --policy-terminal-blend 1.0 --out reports/policy_terminal_n4_eval300
```

The PPO terminal trainer exposes policy/network and optimizer knobs (`--net-arch`, `--activation`, `--learning-rate`, `--n-steps`, `--batch-size`, `--gamma`, `--gae-lambda`, `--ent-coef`, and related PPO settings) so terminal learning experiments can be reproduced without editing code.

Reports compare pure PPO against the same policy routed through ReCoN and list the active mechanisms separately.

For model selection across train seeds:

```bash
uv run python scripts/sweep_policy_terminals.py --train-seeds 550000 560000 570000 --n-poles 4 --dynamics-mode serial_lagrange --dt 0.0005 --action-mode discrete --discrete-action-bins 5 --timesteps 50000 --validation-episodes 100 --final-eval-episodes 300 --out reports/policy_terminal_n4_sweep
```

For target-rung failure/action audits:

```bash
uv run python scripts/audit_failure_actions.py --model-path reports/policy_terminal_n4/ppo_policy_terminal.zip --episodes 80 --dt 0.0005 --dynamics-mode serial_lagrange --failure-offsets 120 80 40 20 10 --probe-horizon 120 --counterfactual-no-noise --out reports/policy_terminal_n4_failure_audit
```

For dt curriculum of the learned terminal:

```bash
uv run python scripts/train_policy_terminal_dt_curriculum.py --dt-values 0.0003 0.0004 0.0005 --n-poles 4 --dynamics-mode serial_lagrange --action-mode discrete --discrete-action-bins 5 --timesteps 25000 --validation-episodes 80 --final-eval-episodes 300 --out reports/policy_terminal_n4_dt_curriculum
```

For validation-aware continuation and a static-vs-terminal oracle bound:

```bash
uv run python scripts/train_policy_terminal_iterative.py --start-model-path reports/policy_terminal_n4/ppo_policy_terminal.zip --n-poles 4 --dynamics-mode serial_lagrange --dt 0.0005 --action-mode discrete --discrete-action-bins 5 --chunk-timesteps 25000 --chunks 4 --validation-episodes 100 --final-eval-episodes 300 --out reports/policy_terminal_n4_iterative
uv run python scripts/analyze_policy_terminal_oracle.py --model-path reports/policy_terminal_n4/ppo_policy_terminal.zip --n-poles 4 --dynamics-mode serial_lagrange --dt 0.0005 --action-mode discrete --discrete-action-bins 5 --episodes 300 --out reports/policy_terminal_n4_oracle
```

## Iterative ReCoN-Native Training

Use `train_until_solved.py` for claim-disciplined training attempts that separate ReCoN-native learning from gain search:

```bash
uv run python scripts/train_until_solved.py --n-poles 3 --target solved_n3 --mode recon_learn_only --budget-episodes 50000 --out reports/train_until_solved_n3
uv run python scripts/train_until_solved.py --n-poles 4 --target solved_n4 --mode recon_learn_only --budget-episodes 50000 --out reports/train_until_solved_n4
```

`recon_learn_only` freezes global proposal gains and allows only ReCoN-owned mechanisms: fast edge plasticity, bandit routing, slow consolidation, and learnable regime/node parameters. The runner writes checkpoints, traces, failure taxonomy, and a report. A run is not solved unless the report status is `solved` and the held-out threshold block passes with the required episode count.

The custom N-link dynamics are a stable control benchmark scaffold, not yet a
validated multibody physics paper. The docs call this out because the research
claim should come from held-out metrics, not a cherry-picked animation.

