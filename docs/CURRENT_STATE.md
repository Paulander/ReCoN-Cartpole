# Current State - 2026-06-10

This note is the handoff point before retuning the project goal. It records the latest honest state of the ReCoN-Cartpole N-link work and the strongest N=4 evidence so far.

## Headline

N=4 is very close, but it is not robustly solved yet. The best current checkpoint clears several independent 300-seed blocks and clears the aggregate 1200-episode success threshold, but one tracked block remains below the per-block success gate. No N=5 solve claim should be made from the current artifacts.

## Best Current N=4 Checkpoint

```text
reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip
```

Environment/evaluation setup used for these metrics:

- `n_poles=4`
- `dynamics_mode=serial_lagrange`
- `dt=0.0005`
- `action_mode=discrete`
- `discrete_action_bins=5`
- `force_mag=10`
- `initial_angle_range=0.05`
- `force_noise=0.02`
- `link_coupling=12`
- `selection_mode=hard_select`
- `policy_terminal_scope=stabilize_chain`
- `policy_observation_mode=normalized_raw`
- `frame_stack=1`

## Held-Out Metrics

| Seed block | Episodes | Mean survival | p10 survival | Success rate | Status |
|---|---:|---:|---:|---:|---|
| `980000..980299` | 300 | 484.15 | 433.6 | 0.6867 | below N=4 success gate |
| `1300000..1300299` | 300 | 485.01 | 435.0 | 0.7067 | passes |
| `1600000..1600299` | 300 | 483.74 | 428.9 | 0.7133 | passes |
| `1700000..1700299` | 300 | 485.82 | 438.9 | 0.7033 | passes |
| Aggregate | 1200 | 484.68 | n/a | 0.7025 | aggregate passes, per-block robustness incomplete |

The project threshold currently used for N=4 is:

- episodes >= 300
- mean survival >= 475
- p10 survival >= 350
- success rate >= 0.70

The aggregate result is encouraging, but because `980000..980299` is still at `0.6867`, the correct claim is **near-solved / not robustly solved**.

## What Is Learned vs Handcoded

The best controller is not pure symbolic ReCoN. It is a learned PPO policy terminal routed through ReCoN:

- Learned: PPO terminal behavior for the `stabilize_chain` proposal.
- ReCoN scaffold: graph routing, `hard_select` regime choice, proposal arbitration, and trace/replay structure.
- Not used as solve evidence: gain-search-only claims, visual demos, or training-set seed performance.

A useful comparison from the best current checkpoint:

- Pure PPO on the same policy under the original `980000..980299` block: success `0.47`.
- ReCoN-routed policy terminal on the same block: success `0.6867`.

So ReCoN routing is materially helping, but the main force policy is still neural/learned.

## Recent Useful Changes

Pushed commits relevant to the current state:

- `2d1f6d0` - normalized raw policy observations.
- `8096232` - training-only success bonus support.
- `7f3b4d9` - iterative curriculum improvements: multiblock validation, score weights, and real hard-seed use.
- `921274e` - training-only terminal failure penalty support.
- `3b0d2da` - desynchronized hard-seed sampling across vectorized workers.

The last item moved the best original held-out success from about `0.6833` to `0.6867` and improved broad validation to `0.7133`.

## Negative Results To Remember

- `soft_select` was worse than `hard_select` for the current learned terminal.
- Lower policy-terminal blend values were worse; `0.9`, `1.0`, and `1.1` were effectively identical due to action quantization.
- Larger `128,128` PPO terminal did not help in the tested run.
- Success bonus and failure penalty are available as training-only tools, but the tested settings did not improve the best checkpoint.
- Combined hard-seed replay was harmful before worker desynchronization and only mildly useful after it.

## Work In Progress At Pause

A hard-seed collection for `980000..980599` was started and then intentionally stopped when pausing the thread. Treat any partial files under `reports/hard_seeds_n4_worker_best_nearmiss_980k_120/` as incomplete unless inspected and regenerated.

The running process was stopped before this note was committed.

## Suggested Next Steps

1. Do not claim N=4 solved yet. Use the current checkpoint as the near-solved baseline.
2. If continuing N=4, collect near-miss failures from a fresh weak block, not from final proof blocks, then evaluate on at least two fresh 300-seed blocks plus the historical `980000` block.
3. Consider an evaluation harness that reports per-block pass/fail and aggregate metrics in one table, so solve claims are less fragile.
4. Once N=4 passes per-block held-out gates consistently, freeze it and run an N=5 probe with the same claim discipline.
5. Keep reports explicit about mechanisms: PPO terminal, ReCoN routing, edge plasticity, bandit persistence, slow consolidation, gain mutation, and hard-seed curriculum.
