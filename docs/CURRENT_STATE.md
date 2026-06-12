# Current State - 2026-06-12

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

The updated project strategy is tracked in [`RECURRENT_TERMINAL_STRATEGY.md`](RECURRENT_TERMINAL_STRATEGY.md). In short: stop adding random knobs; build a fast recurrent/minGRU terminal pipeline, a supervised sequence dataset path, and a fail-fast ladder that tests environment learnability and ReCoN arbitration cleanly.

1. Do not claim N=4 solved yet. Use the current checkpoint as the near-solved baseline.
2. Implement `recon_mingru_terminal` as a ReCoN-arbitrated `ForceProposal`, not a direct action bypass.
3. Add sequence dataset tooling and supervised minGRU training before launching long RL runs.
4. Add a fail-fast recurrent ladder with train/validation/test seed splits and per-block reporting.
5. Once N=4 passes per-block held-out gates consistently, freeze it and run an N=5 probe with the same claim discipline.
6. Keep reports explicit about mechanisms: PPO terminal, minGRU terminal, ReCoN routing, edge plasticity, bandit persistence, slow consolidation, gain mutation, and hard-seed curriculum.

## Tail Iteration Update - 2026-06-12

N=4 remains not solved. Additional PPO, residual, and recurrent experiments improved the evidence map but did not clear the held-out gates.

### PPO Sweep Evidence

Recent bounded PPO continuation slices on the current 5-bin setup:

| Report | Best setting | Episodes | Mean | p10 | Success | Status |
|---|---|---:|---:|---:|---:|---|
| `reports/n4_ppo_sweep_slice_20260612_seed2670k` | lr `5e-7`, clip `0.005`, ent `0.001`, late bonus `0.01`, no VecNormalize | 120 | 486.19 | 437.7 | 0.6833 | not solved |
| `reports/n4_survival_ppo_sweep_20260612_seed2700k` | lr `5e-7`, clip `0.003`, ent `0.001`, late bonus `0.005`, no VecNormalize | 120 | 483.53 | 441.9 | 0.6917 | not solved |
| `reports/n4_ppo_vecnorm_late_slice_20260612_seed2910k` | lr `5e-7`, clip `0.005`, ent `0.0`, no late bonus, no VecNormalize | 60 | 479.02 | 425.8 | 0.5333 | not solved |
| `reports/n4_ppo_vecnorm_true_slice_20260612_seed2911k` | lr `5e-7`, clip `0.003`, ent `0.001`, late bonus `0.005`, VecNormalize | 60 | 471.23 | 410.7 | 0.4500 | not solved |

Interpretation: in this low-learning-rate corner, VecNormalize did not produce a useful breakthrough. The best recent sweep evidence is still the no-VecNormalize survival/late-bonus slice, and even that remains just below the `0.70` success gate.

### Residual Terminal Evidence

The recovery-window residual PPO path is implemented and evaluated through normal ReCoN residual-terminal integration, but it has not improved held-out success:

| Report | Evaluator | Episodes | Mean | p10 | Success | Mean abs delta |
|---|---|---:|---:|---:|---:|---:|
| `reports/n4_recovery_window_residual_ppo_20260612_seed2900k` | unguarded residual | 20 | 485.05 | 467.0 | 0.550 | 5.296 |
| `reports/n4_recovery_window_residual_ppo_guarded_20260612_seed2901k` | guarded residual | 20 | 489.45 | 468.8 | 0.650 | 1.059 |
| `reports/n4_recovery_window_residual_ppo_guarded_20260612_seed2901k` | frozen base comparison | 20 | 489.50 | 468.8 | 0.650 | 0.000 |

Interpretation: the guarded residual mostly preserves the base controller; the unguarded residual damages it. This is useful negative evidence and argues against adding stronger residual authority without better state recognition.

### Recurrent/MinGRU Evidence

The recurrent/minGRU pipeline exists, but the strongest-looking minGRU ladder entries are still too narrow to support a solve claim:

| Report | Evaluation | Episodes | Mean | p10 | Success | Notes |
|---|---|---:|---:|---:|---:|---|
| `reports/n4_recurrent_multiblock_tail_20260612_seed2720k` | ReCoN recurrent PPO final | 120 | 464.99 | 389.5 | 0.4917 | weak |
| `reports/n4_mingru_curriculum_weighted_20260612_seed2813k_ladder_2100k_x20` | ReCoN minGRU ladder | 20 | 488.65 | 464.1 | 0.6000 | too narrow |
| `reports/n4_mingru_incumbent_h256_ladder_4x2_20260612` | ReCoN minGRU ladder | 8 | 500.0 | 500.0 | 1.0000 | smoke-sized, not solve evidence |

Interpretation: recurrent methods are not yet competitive on broad held-out blocks. The high tiny-ladder scores are diagnostic only.

### Observable/Subchain State

The environment observation is cart `x`, cart `x_dot`, and per-pole `sin(theta)`, `cos(theta)`, and `theta_dot`. ReCoN currently exposes one `observe_state_terminal` and one `pole_i_sensor` terminal per pole, not separate terminal nodes for each scalar observable. Learned terminal observation modes include flat normalized raw state and optional adjacent subchain diagnostics (`pair01`, `pair12`, `pair23` deltas/means).

This means the current system does not truly see N=4 recursively as two or three reusable N=2 subproblems. It has subchain features, but they are fed into flat learners rather than shared ReCoN submodules.

### Current Best Next Move

Further blind PPO micro-sweeps have diminishing expected value. The next high-signal implementation step is shared subchain/recurrent composition:

1. Add explicit ReCoN subchain script nodes such as `subchain_0_1`, `subchain_1_2`, and `subchain_2_3`.
2. Give each subchain the same reusable terminal interface over local pair features: delta angle, delta angular velocity, mean angle, mean angular velocity, cart state, and previous force.
3. Train/evaluate the shared subchain terminal through curriculum: N=2/N=3 local stability, N=4 low-angle/no-noise, N=4 current, then hard-seed tail.
4. Keep final N=4 claims tied to held-out mixed blocks only; the current best known N=4 state is still near-solved, not solved.

