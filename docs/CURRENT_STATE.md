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

## Subchain/Recurrent Probe Update - 2026-06-12

After adding explicit adjacent subchain sensor nodes and exposing `normalized_raw4_subchains_prev_force`, two bounded probes were run. Neither solved N=4.

### Flat PPO With Subchain Features

Report: `reports/n4_ppo_subchain_prevforce_slice_20260612_seed2920k`

Setup: feedforward PPO terminal, `normalized_raw4_subchains_prev_force`, one `5000`-step chunk per candidate, no VecNormalize, mixed validation, final held-out blocks `1500000`, `1600000`, and `2100000` with `20` episodes each.

| Grid | clip | ent | late bonus | Mean | p10 | Success | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.003 | 0.0 | 0.0 | 468.73 | 416.2 | 0.4833 | not solved |
| 3 | 0.003 | 0.001 | 0.005 | 470.03 | 407.9 | 0.5000 | not solved |
| 4 | 0.005 | 0.0 | 0.0 | 468.77 | 408.3 | 0.4667 | not solved |
| 7 | 0.005 | 0.001 | 0.005 | 472.55 | 418.1 | 0.4833 | not solved |

Interpretation: simply appending adjacent subchain features to the feedforward PPO terminal did not improve the success tail. This supports the view that the useful next step is not more flat features, but a more compositional/recurrent use of subchains.

### Recurrent Subchain Curriculum Probe

Report: `reports/n4_recurrent_subchain_curriculum_probe_20260612_seed2930k`

Setup: RecurrentPPO terminal with `normalized_raw4_subchains_prev_force`, staged N=3 stable -> N=4 low-angle/no-noise -> N=4 current -> N=4 tail, one small `2048`-step chunk per stage.

| Stage | Mean | p10 | Success | Notes |
|---|---:|---:|---:|---|
| N=3 stable | 500.0 | 500.0 | 1.0000 | warmup only |
| N=4 low-angle/no-noise | 500.0 | 500.0 | 1.0000 | easy-stage transfer works |
| N=4 current narrow validation | 484.06 | 431.0 | 0.7188 | small/narrow validation only |
| Final held-out ReCoN eval | 456.88 | 389.7 | 0.3333 | not robust |

Interpretation: the staged recurrent path can learn easy N=4 and looks promising on a narrow validation set, but that signal did not transfer to the separate final held-out blocks.

### Widened-Validation Recurrent Tail Continuation

Report: `reports/n4_recurrent_subchain_tail_wideval_20260612_seed2935k`

Setup: resumed from the N=4-current recurrent checkpoint above, validated across `900000`, `930000`, `970000`, `1010000`, `1500000`, `1600000`, and `2100000`; two `5000`-step hard-tail chunks.

| Row | Mean | p10 | Success | Promoted |
|---|---:|---:|---:|---|
| start | 472.41 | 412.5 | 0.5536 | yes |
| chunk 1 | 472.10 | 412.0 | 0.5536 | no |
| chunk 2 | 478.30 | 427.5 | 0.6071 | yes |
| final ReCoN eval | 473.30 | 416.6 | 0.5167 | not solved |

Interpretation: widening validation exposed the narrow-validation overestimate, but chunk 2 did improve the wide validation score. The final held-out result is still below the best feedforward PPO terminal results and far below solve threshold. This is a cautiously positive recurrent-tail signal, not a solution.

### Updated Direction

The next recurrent run should use the widened validation starts from the beginning and spend more budget only if intermediate promotions improve both success and p10. The flat subchain PPO path should not be prioritized unless paired with a true shared subchain module or stronger curriculum.

## Tail Continuation And Failure Audit Update - 2026-06-12

Two additional feedforward tail attempts were run against the broad mixed-grid validation set. Neither solved N=4.

### Feedforward Tail Continuation From Sweep Candidate

Report: `reports/n4_feedforward_tail_wideval_20260612_seed2940k`

Setup: resumed from `reports/n4_survival_ppo_sweep_20260612_seed2700k/candidate_01/checkpoint_010000.zip`, validated on 12 seed starts with 10 episodes each, then final-evaluated on `980000`, `1500000`, `1600000`, and `2100000` with 75 episodes each.

| Row | Mean | p10 | CVaR | Success | Promoted |
|---|---:|---:|---:|---:|---|
| start | 484.98 | 437.7 | 420.75 | 0.6500 | yes |
| chunk 1 | 483.71 | 435.0 | 411.83 | 0.6500 | no |
| chunk 2 | 483.73 | 433.9 | 414.58 | 0.6500 | no |
| chunk 3 | 482.13 | 430.9 | 407.50 | 0.6333 | no |
| chunk 4 | 481.25 | 429.8 | 403.83 | 0.6333 | no |

Final held-out ReCoN eval of the preserved best/start checkpoint: mean `482.07`, p10 `428.0`, CVaR `408.9`, success `0.6533` over 300 episodes. This is near-solved but below the configured success threshold.

### Local Failure Action Audits

Reports:

- `reports/n4_failure_action_audit_tailbest_980k_20260612`
- `reports/n4_failure_action_audit_tailbest_1500k_20260612`
- `reports/n4_failure_action_audit_tailbest_1600k_20260612`
- `reports/n4_failure_action_audit_tailbest_980k_early_20260612`
- `reports/n4_failure_action_audit_tailbest_1500k_early_20260612`

The audits found frequent exact-action alternatives near failures, but almost no survival gain from changing one action:

| Seed block | Offset window | Episodes | Success | Main failures | Mistake rate | Mean survival gap |
|---|---|---:|---:|---|---:|---:|
| 980k | 0/2/5/10/20 | 30 | 0.6333 | pole_1, pole_2 | 0.6909 | 0.000 |
| 1500k | 0/2/5/10/20 | 30 | 0.6667 | pole_2, pole_1 | 0.6400 | 0.040 |
| 1600k | 0/2/5/10/20 | 30 | 0.7667 | pole_1, pole_2 | 0.7714 | 0.000 |
| 980k | 40/80/120 | 30 | 0.6333 | pole_1, pole_2 | 0.7576 | 0.000 |
| 1500k | 40/80/120 | 30 | 0.6667 | pole_2, pole_1 | 0.9000 | 0.033 |

Interpretation: the tail failures do not look like simple late one-action mistakes. A residual or action-gate trained only on near-failure windows is unlikely to close the gap by itself.

### Preservation-First Teacher-Anchored Tail Probe

Report directory: `reports/n4_incumbent_teacher_anchor_tail_20260612_seed2950k`

Setup: resumed from `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`, used tiny PPO updates plus teacher-action penalty in low-risk states. The run was interrupted after the first chunk because the teacher-in-env path was too slow for an autonomous iteration loop.

Saved validation evidence:

| Row | Mean | p10 | CVaR | Success | Notes |
|---|---:|---:|---:|---:|---|
| start | 484.98 | 437.7 | 420.75 | 0.6500 | broad mixed-grid start |
| chunk 1 | 483.44 | 434.9 | 417.50 | 0.6333 | latest validation only; regressed |

Interpretation: preservation-first feedforward continuation did not show an early positive signal and was too slow with the current teacher wrapper. The next high-signal path is structural: shared/recurrent subchain composition rather than more flat feedforward tail microfits.

## Shared Subchain Control Hook - 2026-06-12

A first structural ReCoN subchain control hook has been implemented:

- New controller mode: `recon_subchain_terminal`.
- New config: `SubchainBiasConfig` on `RunnerConfig`.
- The hook reuses the same adjacent-pair calculation for each pair `(i, i+1)` and can bias `stabilize_chain` force proposals.
- Diagnostics now include `subchain_bias` with per-pair votes, pair pressure, base force, subchain force, blend, and final proposal force.
- The hook is disabled by default and does not make a solve claim. It is intended as a compositional ReCoN control path for the next curriculum/recurrent experiments.

Verification:

- `uv run pytest -s -q tests/test_controller.py::test_subchain_bias_is_default_off_for_static_recon tests/test_controller.py::test_subchain_bias_mode_changes_stabilize_chain_force_and_reports_votes tests/test_controller.py::test_recon_controller_reports_adjacent_subchain_sensor_values` -> 3 passed.
- `uv run ruff check src/recon_cartpole/recon/engine_runner.py src/recon_cartpole/control/controllers.py tests/test_controller.py` -> passed.
- `uv run pytest -s -q` -> 110 passed.

## Subchain Bias Grid Result - 2026-06-12

The new shared subchain control hook was evaluated on top of the frozen PPO incumbent.

Report: `reports/n4_subchain_bias_grid_20260612_40eps_actiondelta`

Setup: `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`, N=4, 5-bin discrete force, `serial_lagrange`, seed starts `980000`, `1300000`, `1500000`, and `1600000`, 10 episodes per start.

| Candidate | Mean | p10 | Success | Action changes/tick mean | Notes |
|---|---:|---:|---:|---:|---|
| baseline_no_subchain_bias | 481.48 | 444.0 | 0.6250 | 0.0 | best in this probe |
| conservative_default | 475.57 | 421.0 | 0.6250 | 269.2 | changed actions often and degraded p10 |
| low_blend_default | 481.48 | 444.0 | 0.6250 | 0.0 | continuous-force changes did not cross 5-bin boundaries |
| delta_angle_focus | 481.48 | 444.0 | 0.6250 | 0.2 | effectively action-identical |
| mean_pair_focus | 481.48 | 444.0 | 0.6250 | 4.0 | nearly action-identical |
| outer_pair_weighted | 481.48 | 444.0 | 0.6250 | 0.0 | action-identical |

Interpretation: the hook works and is visible in diagnostics, but hand-tuned continuous subchain bias is not enough in the current 5-bin setup. Weak settings rarely change discrete actions; stronger settings damage the tail. This reinforces that the next useful subchain path should be learned/recurrent or action-logit-aware, not static continuous-force nudging.

Implementation note: `StepTrace` and `training.evaluate.rollout(..., trace=True)` now carry `subchain_bias` so future replays can show the pair votes.

