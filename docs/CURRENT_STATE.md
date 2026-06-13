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

## Learned Residual And Recurrent Follow-Up - 2026-06-12

Two follow-up probes were run after the static subchain-bias grid.

### Subchain-Diagnostic Counterfactual Residual Probe

Report: `reports/n4_subchain_counterfactual_residual_probe_20260612_seed2960`

Setup: frozen PPO incumbent, `subchain_diagnostics` residual features, 5 residual bins, option hold `2`, hard-seed collection, short-horizon counterfactual labels, held-out eval on starts `980000`, `1500000`, and `1600000` with 8 episodes each.

Result: the dataset produced no non-noop labels.

| Rows | Non-noop labels | Base mean | Base p10 | Base success | Residual mean | Residual p10 | Residual success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 126 | 0 | 483.5 | 447.2 | 0.625 | 483.5 | 447.2 | 0.625 |

Interpretation: even with subchain diagnostics, pressure shaping, relaxed margin gates, and a 2-step option hold, short-horizon counterfactual probing did not find locally better residual actions. This supports the earlier failure-action audits: the remaining tail is not a simple near-failure one-action rescue problem.

### Widened Recurrent Subchain Curriculum

Report: `reports/n4_recurrent_subchain_wide_curriculum_20260612_seed2970k`

Setup: RecurrentPPO terminal, `normalized_raw4_subchains_prev_force`, N=3 stable -> N=4 low-angle/no-noise -> N=4 current -> hard-tail, widened validation starts from the N=4 current stage onward.

| Stage | Best mean | Best p10 | Best CVaR | Best success | Notes |
|---|---:|---:|---:|---:|---|
| N=3 stable | 500.0 | 500.0 | 500.0 | 1.000 | easy warmup solved |
| N=4 low-angle/no-noise | 500.0 | 500.0 | 500.0 | 1.000 | easy transfer solved |
| N=4 current validation | 484.4 | 435.0 | 408.0 | 0.750 | promising validation only |
| N=4 tail validation | 484.4 | 435.0 | 408.0 | 0.750 | tail chunk not promoted |
| Final held-out ReCoN eval | 471.1 | 394.3 | 375.2 | 0.5625 | not solved |

Interpretation: recurrent curriculum learns the easy distributions and can look good on the mixed validation sample, but the final held-out block still collapses below the feedforward incumbent. This keeps the best known robust N=4 state near-solved but not solved. The next recurrent attempt should either spend materially more budget with wider validation/final blocks from the beginning, or change the recurrent objective/data path rather than relying on another short PPO curriculum slice.

## Motif-Selected Residual Window Update - 2026-06-12

The residual counterfactual collector now accepts an optional motif prototype via `--motif-model-path`. When enabled, failed-episode recovery-window candidates get a `motif_score` from the adjacent-subchain prototype representation and a blended `candidate_rank = motif_rank_weight * motif_score + pressure_rank_weight * recovery_pressure`. This lets future residual runs use learned motif recognition to choose which windows receive counterfactual labels, without hard-coding a recovery action. The report records the motif path and rank weights under `failure_state_selection`.

Focused verification:

- `uv run pytest -s -q tests/test_policy_terminal_training.py::test_counterfactual_residual_can_rank_failure_states_by_motif tests/test_policy_terminal_training.py::test_subchain_motif_prototype_scores_separate_classes tests/test_policy_terminal_training.py::test_subchain_motif_vector_uses_adjacent_pairs` -> 3 passed.
- `uv run ruff check scripts/train_counterfactual_residual_terminal.py tests/test_policy_terminal_training.py` -> passed.

Bounded probes, both evaluated on held-out weak-block seeds and not solve claims:

| run | collection | selector | rows | non-noop labels | base success | residual success | interpretation |
|---|---|---|---:|---:|---:|---:|---|
| `reports/n4_motif_selected_residual_smoke_20260612` | 8 seeds from `2420000` | pure motif top-k | 20 | 0 | 0.625 | 0.625 | motif-selected states had no short-horizon residual advantage |
| `reports/n4_motif_selected_residual_hardpool_20260612` | first 20 hard-pool seeds | pure motif top-k | 84 | 0 | 0.650 | 0.650 | danger motifs were recognizable but not actionable by this residual labeler |
| `reports/n4_motif_pressure_selected_residual_hardpool_20260612` | first 20 hard-pool seeds | motif/pressure blend | 84 | 0 | 0.650 | 0.650 | blended ranking still found no residual advantage in this subset |

Comparison: the older 50-seed pressure-window run `reports/n4_counterfactual_residual_subchain_20260612_seed2390k` produced 170 non-noop labels over 250 rows, so the new selector has not disproven residual learning. It shows that motif-risk predicts near-failure, but pure motif-risk is not equivalent to identifying windows where a simple 5-bin residual action improves short-horizon survival. The next residual attempt should either run the blended selector on the full hard-pool budget or move to trajectory-level recovery-window PPO with motif-score as an observation/filter.

## Motif-Selected Recovery-Window PPO - 2026-06-12

The recovery-window residual PPO script now exposes the same motif-selection controls as the counterfactual residual labeler: `--motif-model-path`, `--motif-score-min`, `--motif-top-k`, `--motif-rank-weight`, and `--pressure-rank-weight`. `windows.json` preserves `motif_score` and `candidate_rank`, and reports record the selector settings. The deployed residual observation shape is unchanged, so trained residual policies remain compatible with normal ReCoN residual-terminal evaluation. Motif risk is used as a recovery-window filter/ranker, not as a hand-authored action rule.

Focused verification:

- `uv run pytest -s -q tests/test_policy_terminal_training.py::test_recovery_window_rows_preserve_motif_selection_metadata tests/test_policy_terminal_training.py::test_counterfactual_residual_can_rank_failure_states_by_motif` -> 2 passed.
- `uv run ruff check scripts/train_recovery_window_residual_policy.py scripts/train_counterfactual_residual_terminal.py tests/test_policy_terminal_training.py` -> passed.

Bounded held-out probe: `reports/n4_motif_recovery_window_residual_ppo_20260612_seed2980`

Setup: frozen incumbent feedforward PPO terminal, 30 hard-pool collection episodes, 136 motif/pressure-ranked recovery windows, `subchain_diagnostics`, 5-bin residual shifts, hold steps 4, 10k PPO timesteps, held-out weak-block eval on `2100000..2100019`.

| evaluator | mean | p10 | cvar | success | mean abs delta | episodes |
|---|---:|---:|---:|---:|---:|---:|
| frozen base | 489.5 | 468.8 | 429.5 | 0.650 | 0.000 | 20 |
| motif recovery-window residual | 485.6 | 457.8 | 419.5 | 0.600 | 2.876 | 20 |

Gate sweep: `reports/n4_motif_recovery_window_residual_ppo_gate_sweep_20260612_seed2980`

| threshold | mean | p10 | cvar | success | interpretation |
|---:|---:|---:|---:|---:|---|
| 0.200 | 485.6 | 457.8 | 419.5 | 0.600 | too many harmful residual changes |
| 0.600 | 489.1 | 468.8 | 428.5 | 0.650 | mostly suppresses harm, no lift |
| 0.900 | 489.4 | 468.8 | 429.5 | 0.650 | matches base-level success, no lift |

Interpretation: motif-selected recovery-window PPO gives a real trajectory-level learner, but the first bounded run learned interventions that hurt unless conservatively gated. This supports the current diagnosis: recognizing dangerous subchain motifs is feasible, but the remaining tail is not solved by simple 5-bin residual shifts trained on short recovery windows. Next attempts should either train a less intrusive residual with stronger behavior-preservation penalties, or move the motif/subchain representation into the primary recurrent policy rather than post-hoc residual overrides. No N=4 solve claim is justified.

## Success-Preservation Recovery Residual Update - 2026-06-12

The recovery-window residual PPO path now supports explicit base-behavior preservation windows from solved collection episodes. Enable with `--preserve-success-stride`; cap per solved episode with `--max-success-preservation-windows`. Preservation windows keep the normal residual observation shape but carry `preserve_success=true` in `windows.json`. During training, pressure-drop shaping is disabled for these windows, requested non-noop residual shifts can be penalized with `--preserve-success-shift-penalty`, and no-op choices can receive `--preserve-success-noop-bonus`. This is intended to stop residual PPO from learning broad, harmful overrides on states the frozen base already handles.

Focused verification:

- `uv run pytest -s -q tests/test_policy_terminal_training.py::test_recovery_window_rows_preserve_motif_selection_metadata tests/test_policy_terminal_training.py::test_recovery_window_rows_can_add_success_preservation_windows` -> 2 passed.
- `uv run ruff check scripts/train_recovery_window_residual_policy.py tests/test_policy_terminal_training.py` -> passed.

Bounded held-out probe: `reports/n4_preserve_motif_recovery_window_residual_ppo_20260612_seed2981`

Setup: frozen incumbent feedforward PPO terminal, 40 hard-pool collection episodes, 176 recovery windows plus 72 success-preservation windows, motif/pressure-ranked recovery selection, `subchain_diagnostics`, 5-bin residual shifts, hold steps 4, 10k PPO timesteps, held-out weak-block eval on `2100000..2100019`.

| evaluator | mean | p10 | cvar | success | mean abs delta | episodes |
|---|---:|---:|---:|---:|---:|---:|
| frozen base | 489.5 | 468.8 | 429.5 | 0.650 | 0.000 | 20 |
| preservation residual | 484.7 | 458.9 | 420.0 | 0.500 | 3.352 | 20 |

Gate sweep: `reports/n4_preserve_motif_recovery_window_residual_ppo_gate_sweep_20260612_seed2981`

| threshold | mean | p10 | cvar | success | interpretation |
|---:|---:|---:|---:|---:|---|
| 0.200 | 484.7 | 458.9 | 420.0 | 0.500 | residual remains too intervention-heavy |
| 0.400 | 488.6 | 468.8 | 426.5 | 0.650 | suppresses most harm, no lift |
| 0.900 | 489.4 | 468.8 | 429.0 | 0.650 | matches base-level success, no lift |

Interpretation: adding solved-episode preservation windows was the right guardrail, but this PPO residual formulation still learned harmful residual shifts. The residual path is now better instrumented, yet the evidence argues against spending more short-run budget on post-hoc residual overrides. The higher-signal next path is to put subchain/motif information into the primary recurrent policy or to train a much more conservative residual objective with explicit no-op class imbalance. No N=4 solve claim is justified.


## Motif Observable for Primary Recurrent Policy - 2026-06-12

Current answer to the observable question: the controller has raw/padded observation modes for cart position, cart velocity, pole angles, pole angular velocities, previous force, and adjacent-subchain feature modes. It does not yet have a separate learned ReCoN terminal for every primitive observable. The latest change adds a learned motif-score observable that can be fed directly into the primary minGRU terminal. The score is derived from the existing adjacent-subchain success/failure prototype model, so it is a causal policy input rather than a report-only gate.

What this enables: successful-seed and failing-seed local patterns can be recognized online and used by the recurrent policy during supervised/curriculum training and evaluation. This is not a hand-coded recovery action and it is not a solve claim; it is a compact learned feature that can help the policy notice states similar to earlier success/failure motifs on held-out seeds.

Recursive/subchain status: the system still does not recursively execute N=4 as two explicit N=2 ReCoN programs. It now exposes adjacent-link motif information in a reusable way, which is the practical first step toward that structure. The next stronger version would add explicit per-subchain terminals/proposals and an arbiter that composes their advice across overlapping two-link windows.

Implementation notes:

- Added `src/recon_cartpole/control/motif_features.py` for loading prototype models, building adjacent-subchain motif vectors, and scoring online states.
- `MinGRUTerminalConfig` now supports `include_motif_score`, `motif_model_path`, and `motif_score_scale`; saved checkpoints persist those fields.
- `build_policy_dataset.py` stores `motif_scores`; supervised minGRU training can append them to inputs; curriculum aggregation remains backward-compatible with old datasets by filling missing motif scores with zero.
- `train_mingru_curriculum.py` and `train_recurrent_terminal_ladder.py` can pass motif-score observables through training/evaluation configs.

Verification:

- `uv run pytest tests/test_policy_terminal_training.py -q -s` -> 64 passed.
- `uv run ruff check src/recon_cartpole/control/motif_features.py src/recon_cartpole/recon/mingru_terminal.py scripts/build_policy_dataset.py scripts/train_mingru_supervised.py scripts/train_mingru_curriculum.py scripts/train_recurrent_terminal_ladder.py tests/test_policy_terminal_training.py` -> passed.
- Integration smoke: `reports/smoke_mingru_motif_curriculum_20260612`, horizon 120, 4 eval seeds, motif score enabled, success 1.0. This is only a wiring smoke, not a robust N=4 result.

No N=4 solve claim is justified from this change. The immediate next experiment is a real motif-enabled recurrent curriculum on the normal horizon/held-out blocks, compared against the same config without motif score.


## Motif Recurrent DAgger Batch - 2026-06-12

A full-horizon paired recurrent curriculum batch tested whether the new learned motif-score observable helps the primary minGRU terminal on held-out N=4 mixed blocks. All rows below use the same 80 held-out eval seeds from starts `1900000`, `2000000`, `2100000`, and `2200000` with 20 episodes per block. These are not train seeds and are not solve claims.

| run | key change | samples | pure mean | pure p10 | pure success | ReCoN mean | ReCoN p10 | ReCoN success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `reports/n4_mingru_curriculum_subchain_nomotif_20260612_seed3210k` | subchain curriculum, no motif scalar | 39648 | 462.9 | 389.4 | 0.400 | 464.7 | 389.6 | 0.425 |
| `reports/n4_mingru_curriculum_subchain_motif_20260612_seed3210k` | same curriculum plus motif scalar | 39648 | 462.6 | 389.4 | 0.400 | 464.5 | 389.4 | 0.425 |
| `reports/n4_mingru_curriculum_subchain_motif_dagger2_20260612_seed3610k` | student hard-tail rollouts, teacher labels, resume from motif | 38668 | 456.1 | 354.0 | 0.4875 | 483.0 | 443.8 | 0.5875 |
| `reports/n4_mingru_curriculum_subchain_motif_dagger3_20260612_seed4010k` | larger student-tail set, stronger tail weight | 42343 | 484.2 | 423.7 | 0.675 | 486.5 | 442.8 | 0.6625 |
| `reports/n4_mingru_curriculum_subchain_motif_dagger4_20260612_seed4410k` | larger tail set, lower LR, stronger low-return weighting | 53447 | 487.0 | 449.9 | 0.675 | 487.0 | 449.9 | 0.675 |
| `reports/n4_mingru_curriculum_subchain_motif_dagger5_20260612_seed4810k` | conservative follow-up from DAgger4 | 53507 | 487.4 | 443.8 | 0.675 | 487.4 | 443.8 | 0.675 |

Interpretation: the motif scalar alone did not move the recurrent policy. The large improvement came from DAgger-style hard-tail collection where the student recurrent policy generated drift states and the frozen teacher supplied labels. This directly supports the compounding-error diagnosis: plain supervised imitation can show high validation action accuracy while failing in rollout. DAgger3/4/5 pushed held-out success from 0.425 to 0.675 and p10 from about 389 to about 450, but the batch plateaued below 0.70 on this 80-seed block. No N=4 solve claim is justified.

Engineering fix found during this batch: `MinGRUTerminal.load_checkpoint` was overwriting runtime arbitration knobs from saved checkpoint config, which made passthrough/arbitration sweeps unreliable. Runtime fields such as `blend`, `scope`, `confidence_floor`, and passthrough thresholds are now preserved from the caller while architecture/input fields still load from the checkpoint.

Verification for the code fix:

- `uv run pytest tests/test_policy_terminal_training.py -q -s` -> 65 passed.
- `uv run ruff check src/recon_cartpole/recon/mingru_terminal.py tests/test_policy_terminal_training.py` -> passed.

Next best move: stop adding motif scalars and continue with distribution-aware learning. The strongest path is either another DAgger/on-policy loop that explicitly mines the remaining held-out failures, or a true recurrent PPO/fine-tuning step initialized from DAgger4/5. The current supervised DAgger loop appears useful but is flattening just under the target band.


## Recurrent Failure-Mining DAgger Branch - 2026-06-13

Added `scripts/collect_mingru_hard_seeds.py` to mine fresh hard seeds for trained minGRU recurrent terminals. The collector supports `pure_mingru_policy` and `recon_mingru_terminal`, records per-seed survival/failure taxonomy, and writes `hard_seeds.json`, `hard_seeds.txt`, and `hard_seeds.md`. This makes recurrent DAgger tail selection reproducible instead of hand-copying failure lists.

Fresh mining run: `reports/n4_mingru_dagger5_hardseed_mine_20260613_seed5200k`

Setup: DAgger5 checkpoint, ReCoN-wrapped minGRU, N=4 serial Lagrange, 5-bin actions, horizon 500, fresh seeds `5200000..5200399`, min hard survival 350, not part of the 80-seed held-out reporting block.

| scan | episodes | mean | p10 | success | hard seeds | failure counts |
|---|---:|---:|---:|---:|---:|---|
| DAgger5 fresh mining block | 400 | 484.1 | 435.0 | 0.710 | 116 | `pole_1_angle`: 66, `pole_2_angle`: 50 |

Targeted refinements evaluated on the unchanged 80 held-out mixed seeds from starts `1900000`, `2000000`, `2100000`, and `2200000`:

| run | tail data | samples | pure mean | pure p10 | pure success | ReCoN mean | ReCoN p10 | ReCoN success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `reports/n4_mingru_curriculum_subchain_motif_dagger5_20260612_seed4810k` | previous broad student-tail DAgger | 53507 | 487.4 | 443.8 | 0.675 | 487.4 | 443.8 | 0.675 |
| `reports/n4_mingru_curriculum_subchain_motif_mined_dagger6_20260613_seed5200k` | 116 mined failure/near-miss seeds only | 67015 | 486.0 | 439.7 | 0.675 | 486.0 | 439.7 | 0.675 |
| `reports/n4_mingru_curriculum_subchain_motif_balanced_dagger7_20260613_seed5200k` | 116 mined failures plus 116 success-preservation seeds | 125086 | 486.8 | 444.9 | 0.675 | 486.8 | 444.9 | 0.675 |

Interpretation: fresh-seed mining confirms the current DAgger5 recurrent policy can exceed 0.70 on one independent 400-seed block, but targeted training on that mined pool did not improve the standing 80-seed held-out block. Mined-only DAgger slightly hurt mean/p10; balanced hard/success preservation recovered p10 but still plateaued at 0.675. More supervised DAgger with the same architecture/objective is unlikely to crack the remaining gap by itself.

Next best move: use DAgger5 or DAgger7 as initialization for a true on-policy fine-tuning objective, or switch to explicit residual/on-policy learning over the remaining failure taxonomy. The persistent failures are now mostly `pole_1_angle` and `pole_2_angle`, so future training should optimize late failure recovery directly rather than only imitating teacher labels on more collected states. No N=4 solve claim is justified from this branch.

Verification for the collector code:

- `uv run ruff check scripts/collect_mingru_hard_seeds.py tests/test_policy_terminal_training.py` -> passed.
- `uv run pytest tests/test_policy_terminal_training.py::test_recurrent_terminal_scripts_import_and_hash_configs -q -s` -> passed.


## minGRU On-Policy Fine-Tuning Baseline - 2026-06-13

Added `scripts/train_mingru_onpolicy.py`, a conservative actor-critic fine-tuner for minGRU recurrent checkpoints. It rolls out the current recurrent policy on fresh seeds, optimizes discounted survival returns with value loss and entropy, and applies a frozen-reference KL penalty to reduce drift from the DAgger checkpoint. This is not full PPO yet; it is a small on-policy baseline to test whether survival-gradient updates can move the recurrent tail.

Verification and smoke:

- `uv run ruff check scripts/train_mingru_onpolicy.py tests/test_policy_terminal_training.py` -> passed.
- `uv run pytest tests/test_policy_terminal_training.py::test_recurrent_terminal_scripts_import_and_hash_configs tests/test_policy_terminal_training.py::test_mingru_onpolicy_discounted_returns -q -s` -> 2 passed.
- `reports/smoke_mingru_onpolicy_20260613` completed successfully on a short horizon; this was only an integration smoke.

Held-out eval uses the same 80 mixed seeds from starts `1900000`, `2000000`, `2100000`, and `2200000` as the DAgger plateau reports.

| run | train seeds | update style | pure mean | pure p10 | pure success | ReCoN mean | ReCoN p10 | ReCoN success | interpretation |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `reports/n4_mingru_curriculum_subchain_motif_dagger5_20260612_seed4810k` | supervised DAgger baseline | teacher-label DAgger | 487.4 | 443.8 | 0.675 | 487.4 | 443.8 | 0.675 | standing recurrent plateau |
| `reports/n4_mingru_onpolicy_from_dagger5_20260613_seed6300k` | 96 fresh seeds | LR `1e-6`, KL `0.05` | 487.4 | 443.8 | 0.675 | 486.2 | 440.6 | 0.6625 | too conservative to improve pure policy; wrapper dipped |
| `reports/n4_mingru_onpolicy_stronger_from_dagger5_20260613_seed6500k` | 48 fresh seeds | LR `5e-6`, KL `0.01` | 487.4 | 443.9 | 0.675 | 486.2 | 440.6 | 0.6625 | stronger updates still did not improve held-out block |

Interpretation: this first on-policy minGRU actor-critic baseline did not crack the DAgger plateau. The pure recurrent policy barely moved, while ReCoN-wrapped performance dipped from 0.675 to 0.6625. The likely issue is that single-policy-gradient updates over sparse survival returns are too noisy and do not target the specific late `pole_1_angle` / `pole_2_angle` failure modes sharply enough.

Next best move: either implement a more faithful PPO-style clipped objective with stored old log-probs and minibatch epochs, or return to the residual/on-policy specialist path where the action space is deliberately small and the objective can focus on late failures without rewriting the whole recurrent controller. No N=4 solve claim is justified from these runs.

## N=4 Tail PPO/Residual Continuation Update - 2026-06-13

Two additional bounded runs tested whether the current best 5-bin feedforward PPO terminal can be nudged over the N=4 held-out threshold without hardcoded action rules.

Residual specialist run: `reports/n4_survival_base_residual_guarded_20260613_seed6600k`

Setup: frozen base `reports/n4_survival_ppo_sweep_20260612_seed2700k/candidate_01/checkpoint_010000.zip`, `subchain_diagnostics` residual features, 3-bin residual action shifts, risk gate `0.55`, hold steps `2`, collection seeds from `2100000`, and held-out final eval on starts `1500000` and `1600000` with 60 episodes each.

| evaluator | mean | p10 | cvar | success | episodes | mean abs residual delta |
|---|---:|---:|---:|---:|---:|---:|
| frozen survival PPO base | 483.5 | 441.9 | 411.8 | 0.6917 | 120 | 0.000 |
| guarded residual terminal | 483.5 | 441.9 | 411.8 | 0.6917 | 120 | 0.330 |

The guarded residual changed only one held-out episode, worsening it by one tick. This is useful negative evidence: the conservative residual path is mostly inert, while earlier less guarded residual runs over-intervened and hurt success. The remaining gap does not look like a simple one-step residual rescue problem under the current features/action bins.

Tiny PPO continuation run: `reports/n4_survival_tail_continue_tiny_20260613_seed6710k`

Setup: resumed from the same survival PPO checkpoint, trained on the previous validation-tail seed pool with tiny PPO updates (`2500` steps x 4 chunks, LR `2.5e-7`, clip `0.002`), selected by mixed validation starts `900000`, `930000`, `970000`, `1010000`, `1040000`, `1070000`, `1140000`, and `1300000`, then final-evaluated on starts `1500000` and `1600000`.

| checkpoint | validation success | validation p10 | final success | final p10 | interpretation |
|---|---:|---:|---:|---:|---|
| start / prior best | 0.6750 | 437.7 | 0.6917 | 441.9 | incumbent |
| chunk 4 promoted on validation | 0.6833 | 434.9 | 0.6833 | 441.9 | validation lift did not transfer |

Compared with the prior best on the same final seeds, the promoted checkpoint changed 6 episodes: 2 improved slightly and 4 worsened, including one prior success at seed `1600042` failing at 471 steps. This confirms the 5-bin feedforward PPO terminal is in a narrow plateau where tiny training changes can trade held-out successes rather than adding robust capacity.

A teacher-anchored continuation attempt, `reports/n4_survival_tail_teacher_anchor_20260613_seed6720k`, was interrupted after the start validation because the current teacher-in-env implementation did not finish even one 2500-step chunk in practical loop time. Before retrying that branch, teacher actions should be cached or the run should use a cheaper dummy-env configuration.

Current N=4 state: near-solved, not solved. The strongest 5-bin feedforward evidence remains the survival PPO checkpoint at held-out success `0.6917` on the 120-episode `1500000`/`1600000` block, below the configured `0.70` success threshold and below the broader 300-episode solve discipline. The most promising next step is structural rather than another micro-sweep: make the adjacent subchain view a shared learned terminal/module, so N=4 can be controlled compositionally as overlapping reusable N=2-style local problems plus a global ReCoN arbiter.

## minGRU PPO-Style Fine-Tuning Probe - 2026-06-13

Added `scripts/train_mingru_ppo.py`, a clipped PPO-style recurrent fine-tuner for minGRU terminal checkpoints. Unlike the earlier actor-critic baseline, it stores rollout sequences, actions, old log-probs, old values, discounted returns, and normalized advantages, then applies PPO clipped minibatch updates with optional frozen-reference KL preservation. The script reports active mechanisms separately and evaluates pure minGRU versus ReCoN-routed minGRU on held-out seeds.

Verification and smoke:

- `uv run ruff check scripts/train_mingru_ppo.py tests/test_policy_terminal_training.py` -> passed.
- `uv run pytest tests/test_policy_terminal_training.py::test_recurrent_terminal_scripts_import_and_hash_configs tests/test_policy_terminal_training.py::test_mingru_ppo_clipped_policy_loss_prefers_clipped_surrogate tests/test_policy_terminal_training.py::test_mingru_ppo_normalize_can_be_disabled -q -s` -> 3 passed.
- `reports/smoke_mingru_ppo_20260613` completed a short-horizon integration smoke.

Held-out eval uses the same 80 mixed seeds from starts `1900000`, `2000000`, `2100000`, and `2200000` as the recurrent DAgger/on-policy reports.

| run | start checkpoint | train seeds | update style | pure mean | pure p10 | pure success | ReCoN mean | ReCoN p10 | ReCoN success | interpretation |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `reports/n4_mingru_curriculum_subchain_motif_dagger5_20260612_seed4810k` | supervised baseline | DAgger | teacher-label DAgger | 487.4 | 443.8 | 0.675 | 487.4 | 443.8 | 0.675 | baseline |
| `reports/n4_mingru_ppo_from_dagger5_20260613_seed7100k` | DAgger5 | 48 mined hard seeds | LR `2e-6`, clip `0.03`, KL `0.05` | 487.4 | 443.9 | 0.675 | 486.2 | 440.6 | 0.6625 | too conservative; wrapper dipped |
| `reports/n4_mingru_curriculum_subchain_motif_balanced_dagger7_20260613_seed5200k` | supervised baseline | DAgger | balanced hard/success DAgger | 486.8 | 444.9 | 0.675 | 486.8 | 444.9 | 0.675 | baseline |
| `reports/n4_mingru_ppo_from_dagger7_mixed_20260613_seed7200k` | DAgger7 | 64 fresh mixed seeds | LR `8e-6`, clip `0.08`, KL `0.005` | 486.7 | 445.0 | 0.675 | 486.4 | 442.4 | 0.6625 | sampled rollout stats improved, held-out did not |

Interpretation: PPO-style recurrent fine-tuning is now implemented and working, but these first bounded runs did not crack the N=4 held-out plateau. The stronger DAgger7 run improved sampled training rollouts from 0.4375 to 0.75/0.8125 success during collection, while frozen-reference KL stayed tiny, but held-out success remained 0.675 for pure minGRU and dipped to 0.6625 through the ReCoN wrapper. This suggests the current minGRU terminal/action interface is not adding robust capacity by small on-policy updates alone.

Next best move: stop spending most runtime on small updates to the same flat global recurrent terminal. The evidence now points more strongly toward explicit shared adjacent-subchain composition: train a reusable pair/local terminal over `0-1`, `1-2`, and `2-3` features and let ReCoN arbitrate/globalize those local proposals, rather than feeding subchain diagnostics into one global policy head.

## Learned Shared Subchain Terminal Slice - 2026-06-13

Added a structural ReCoN subchain path instead of another global policy micro-update:

- New controller mode/config path: `recon_learned_subchain_terminal` plus `learned_subchain_terminal=SubchainTerminalConfig(...)`.
- New module: `src/recon_cartpole/recon/subchain_terminal.py`.
- New trainer: `scripts/train_subchain_pair_terminal.py`.

The learned terminal applies one shared MLP to every adjacent pair (`0-1`, `1-2`, `2-3` for N=4). Each pair sees cart state, local delta angle/velocity, local mean angle/velocity, and optional pair-position. ReCoN aggregates confidence/pressure-weighted pair force votes and blends the resulting local subchain force into the active proposal. Traces now expose `learned_subchain_terminal` diagnostics with per-pair votes, weights, confidence, pressure, base force, subchain force, and final proposal force.

Verification and smoke:

- `uv run ruff check src/recon_cartpole/recon/subchain_terminal.py src/recon_cartpole/recon/engine_runner.py src/recon_cartpole/control/controllers.py scripts/train_subchain_pair_terminal.py tests/test_controller.py tests/test_policy_terminal_training.py` -> passed.
- `uv run pytest tests/test_controller.py::test_learned_subchain_terminal_mode_changes_force_and_reports_votes tests/test_policy_terminal_training.py::test_recurrent_terminal_scripts_import_and_hash_configs -q -s` -> 2 passed.
- `reports/smoke_subchain_pair_terminal_20260613` completed a short low-angle integration smoke.

Bounded N=4 probe: `reports/n4_subchain_pair_distill_probe_20260613_seed8300k`

Setup: distilled the frozen survival PPO teacher (`reports/n4_survival_ppo_sweep_20260612_seed2700k/candidate_01/checkpoint_010000.zip`) into the shared pair terminal from 16 full N=4 current-distribution teacher episodes, then evaluated on held-out starts `1500000` and `1600000` with 20 episodes per block. The learned terminal was enabled as a conservative augmentation with blend `0.10`.

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| frozen PPO ReCoN base | 485.6 | 443.7 | 0.700 | 40 |
| base + learned shared subchain terminal | 485.6 | 443.7 | 0.700 | 40 |

Interpretation: the structural path is implemented and traceable, and the first conservative distillation probe did not damage the PPO base on this small held-out slice. It also did not add lift; simple teacher-force distillation appears mostly action-neutral at low blend. This is not a solve claim because the evaluation is only 40 episodes and reproduces the base on the same slice. The next useful experiment is not more plain distillation, but training the pair terminal on counterfactual/local recovery labels or residual advantages so pair votes can differ from the global teacher in the failure tail.

## Counterfactual Shared Subchain Pair Labels - 2026-06-13

Extended `scripts/train_subchain_pair_terminal.py` with `--label-mode counterfactual_recovery`. The mode rolls out the frozen teacher, samples near-failure windows plus optional success-preservation windows, probes all 5 discrete forces for a short local option, and trains the shared adjacent-pair terminal on either a better local recovery force or a preserve/no-op target. Training rows now carry sample weights and source labels, and reports separate teacher distillation from counterfactual recovery labels in their active mechanisms.

Verification and smoke:

- `uv run ruff check scripts/train_subchain_pair_terminal.py tests/test_policy_terminal_training.py` -> passed.
- `uv run pytest tests/test_policy_terminal_training.py::test_recurrent_terminal_scripts_import_and_hash_configs -q -s` -> passed.
- `reports/smoke_subchain_pair_counterfactual_20260613` completed a short low-horizon integration smoke.

Bounded N=4 probe: `reports/n4_subchain_pair_counterfactual_probe_20260613_seed8500k`

Setup: frozen survival PPO teacher (`reports/n4_survival_ppo_sweep_20260612_seed2700k/candidate_01/checkpoint_010000.zip`), 24 current-distribution collection episodes, near-failure offsets `[0, 2, 5, 10, 20, 40, 80]`, conservative blend `0.12`, and held-out starts `1500000` and `1600000` with 20 episodes per block.

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| frozen PPO ReCoN base | 485.6 | 443.7 | 0.700 | 40 |
| base + counterfactual shared subchain terminal | 485.6 | 443.7 | 0.700 | 40 |

Dataset source counts were `counterfactual_no_better: 189` and `preserve_success: 135`; there were no `counterfactual_recovery` rows under this probe. Interpretation: the counterfactual label plumbing works and remains claim-disciplined, but this specific short-horizon local force search did not discover better-than-base recovery actions. It is neutral evidence, not a solve. The next performance move should therefore be either a stronger local option search or another PPO continuation slice, not more identical subchain distillation.

## Targeted PPO Continuation Slice - 2026-06-13

Ran a 4-candidate continuation sweep from the current best 5-bin survival PPO terminal (`reports/n4_survival_ppo_sweep_20260612_seed2700k/candidate_01/checkpoint_010000.zip`). The slice kept the current setup fixed (`N=4`, `serial_lagrange`, `dt=0.0005`, 5 discrete action bins, force noise `0.02`, hard-select ReCoN policy terminal) and tested tiny PPO continuation updates around the incumbent: LR `2.5e-7`, clip `0.002/0.004`, one epoch, late-survival bonus `0.005/0.01`, 20k max continuation steps per candidate. Validation used the mixed held-out starts `900000`, `930000`, `970000`, `1010000`, `1040000`, `1070000`, `1140000`, and `1300000`; final eval used starts `1500000` and `1600000` with 60 episodes each.

Run: `reports/n4_ppo_targeted_continue_20260613_seed8800k`

| candidate | LR | clip | late bonus | best validation success | final mean | final p10 | final cvar | final success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.5e-7 | 0.002 | 0.005 | 0.6833 | 483.3 | 441.9 | 411.8 | 0.6917 |
| 1 | 2.5e-7 | 0.002 | 0.010 | 0.6833 | 483.5 | 441.9 | 411.8 | 0.6917 |
| 2 | 2.5e-7 | 0.004 | 0.005 | 0.6833 | 483.4 | 441.9 | 411.8 | 0.6917 |
| 3 | 2.5e-7 | 0.004 | 0.010 | 0.6833 | 483.4 | 441.9 | 411.8 | 0.6917 |

Interpretation: this sweep did not solve N=4 and did not beat the incumbent final success. All four candidates could trade validation up from `0.6750` to `0.6833`, but final held-out success remained `0.6917` with unchanged p10/cvar. This is useful negative evidence: the current 5-bin feedforward PPO terminal is not failing because the last continuation LR/clip/late-bonus setting is slightly off. The next attempt should change structure or information, for example a stronger learned local/subchain option search, a residual trained from longer counterfactual options, or a recurrent/cached-teacher curriculum that adds useful state memory without perturbing solved seeds.

## Long-Option Counterfactual Residual Labels - 2026-06-13

Extended `scripts/train_counterfactual_residual_terminal.py` so counterfactual residual labels can score multi-tick options instead of single near-instant perturbations. The simulator now asks the frozen ReCoN/PPO controller for the base action at each forced tick and applies the residual shift relative to that base action, matching the real residual-terminal integration more closely. New knobs: `--option-tail-steps` and `--tail-shift-penalty`. With `--option-hold-steps 20 --option-tail-steps 20`, the labeler can discover sustained “push, then tail/no-op” interventions that were invisible under 2-tick probes at `dt=0.0005`.

Verification:

- `uv run ruff check scripts/train_counterfactual_residual_terminal.py tests/test_policy_terminal_training.py` -> passed.
- `uv run pytest tests/test_policy_terminal_training.py::test_counterfactual_residual_builds_two_phase_option_sequences tests/test_policy_terminal_training.py::test_counterfactual_residual_label_state_respects_advantage_gates tests/test_policy_terminal_training.py::test_counterfactual_residual_label_state_can_use_pressure_advantage tests/test_policy_terminal_training.py::test_recurrent_terminal_scripts_import_and_hash_configs -q -s` -> 4 passed.
- `reports/smoke_counterfactual_residual_twophase_20260613` completed a short integration smoke.

Bounded N=4 long-option probe: `reports/n4_counterfactual_residual_longoption_probe_20260613_seed8910k`

Setup: frozen survival PPO base, `subchain_diagnostics` residual features, 5 residual bins, 20-tick residual hold, 20-tick tail option search, hard-seed collection from `reports/hard_seeds_n4_combined_nearmiss_600/hard_seeds.json`, failure windows biased to 20-180 ticks before failure, and held-out eval on starts `1500000` and `1600000` with 20 episodes per block.

| item | value |
|---|---:|
| rows | 31 |
| non-noop labels | 7 |
| max score gap | 0.976 |
| mean score gap | 0.219 |
| non-noop recall | 0.857 |

| evaluator | mean | p10 | cvar | success | mean abs residual delta |
|---|---:|---:|---:|---:|---:|
| frozen PPO ReCoN base | 485.6 | 443.7 | 423.8 | 0.700 | 0.000 |
| long-option residual, gate 0.65 | 485.5 | 443.6 | 422.5 | 0.700 | 0.306 |

Gate sweep: `reports/n4_counterfactual_residual_longoption_gate_sweep_20260613`

| gate | mean | p10 | cvar | success |
|---:|---:|---:|---:|---:|
| 0.65 | 485.5 | 443.6 | 422.5 | 0.700 |
| 0.75 | 485.5 | 443.6 | 422.8 | 0.700 |
| 0.85 | 485.6 | 443.7 | 423.5 | 0.700 |
| 0.95 | 485.6 | 443.7 | 423.3 | 0.700 |
| 1.05 | 485.6 | 443.7 | 423.8 | 0.700 |

Interpretation: this fixes an important learning-signal problem. The earlier residual probes were too short to affect the physics at `dt=0.0005`; long options finally produce real non-noop labels. However, the first learned residual still does not improve held-out N=4 and slightly hurts tail metrics when allowed to intervene. Higher gates simply suppress it back to the base. The next residual attempt should use more diverse positive windows plus success-preservation rows, and should probably train/evaluate a confidence/gating head instead of relying only on the scalar risk gate.


### Broader Long-Option Residual Probe

Run: `reports/n4_counterfactual_residual_longoption_broad_20260613_seed8920k`

This repeated the long-option residual setup with 16 hard/near-miss collection seeds, wider 20-200 tick pre-failure windows, success-preservation rows from solved collection episodes, and a 120-episode held-out final eval on starts `1500000` and `1600000`.

| item | value |
|---|---:|
| rows | 78 |
| non-noop labels | 13 |
| max score gap | 0.975 |
| mean score gap | 0.160 |
| non-noop recall | 0.769 |

| evaluator | mean | p10 | cvar | success | mean abs residual delta |
|---|---:|---:|---:|---:|---:|
| frozen PPO ReCoN base | 483.5 | 441.9 | 411.8 | 0.6917 | 0.000 |
| broad long-option residual, gate 0.75 | 483.2 | 441.0 | 409.9 | 0.6833 | 0.425 |

Interpretation: the broader run made the residual more active but reduced held-out success by one episode and degraded p10/cvar. So the bottleneck has moved: long-option counterfactual labels now create a learnable signal, but the policy does not know when to abstain. The next residual iteration should add explicit confidence/gating supervision or a two-head residual (`action`, `apply/no-apply`) trained with stronger success-preservation negatives, rather than only raising/lowering the scalar risk gate.

### Gated Long-Option Residual Probe

Implemented a two-head counterfactual residual terminal format: one head predicts the residual class, and a separate apply head predicts whether the residual should be allowed to change the frozen PPO action. Old residual `.pt` checkpoints remain backward-compatible; new gated checkpoints save `apply_state_dict` and report `apply_probability`, `apply_threshold`, and `apply_allowed` in the ReCoN policy-terminal trace.

Verification:

- `uv run ruff check src/recon_cartpole/recon/engine_runner.py scripts/train_counterfactual_residual_terminal.py tests/test_controller.py tests/test_policy_terminal_training.py` -> passed.
- `uv run pytest tests/test_controller.py::test_torch_residual_policy_terminal_can_be_loaded tests/test_controller.py::test_torch_gated_residual_policy_terminal_reports_apply_probability tests/test_controller.py::test_residual_policy_terminal_apply_gate_can_suppress_shift tests/test_controller.py::test_residual_policy_terminal_bin_delta_can_hold_option_across_ticks tests/test_policy_terminal_training.py::test_counterfactual_residual_train_model_oversamples_non_noop_labels tests/test_policy_terminal_training.py::test_counterfactual_residual_train_model_can_emit_apply_gate tests/test_policy_terminal_training.py::test_counterfactual_residual_builds_two_phase_option_sequences -q -s` -> 7 passed.
- `reports/smoke_gated_counterfactual_residual_20260613` completed a gated integration smoke.

Bounded gated probe: `reports/n4_gated_counterfactual_residual_probe_20260613_seed8930k`

Setup: frozen survival PPO base, `subchain_diagnostics` residual features, 5 residual bins, 20-tick residual hold, 20-tick tail option search, 8 hard/near-miss collection seeds, success-preservation negatives from solved collection episodes, and held-out eval on starts `1500000` and `1600000` with 20 episodes per block.

| item | value |
|---|---:|
| rows | 41 |
| non-noop labels | 3 |
| apply labels after oversampling | 12 positive / 38 negative |
| action train accuracy | 0.740 |
| non-noop recall | 0.667 |
| apply accuracy | 0.880 |
| max score gap | 0.964 |

| evaluator | mean | p10 | cvar | success | mean abs residual delta |
|---|---:|---:|---:|---:|---:|
| frozen PPO ReCoN base | 485.6 | 443.7 | 423.8 | 0.700 | 0.000 |
| gated residual, apply threshold 0.55 | 485.5 | 443.5 | 423.0 | 0.700 | 0.678 |

Apply-threshold sweep: `reports/n4_gated_counterfactual_residual_probe_20260613_seed8930k/apply_gate_sweep.json`

| apply threshold | mean | p10 | cvar | success | mean abs residual delta |
|---:|---:|---:|---:|---:|---:|
| 0.55 | 485.5 | 443.5 | 423.0 | 0.700 | 0.678 |
| 0.65 | 485.5 | 443.5 | 422.3 | 0.700 | 0.389 |
| 0.75 | 485.4 | 443.5 | 422.0 | 0.700 | 0.156 |
| 0.85 | 485.6 | 443.6 | 423.5 | 0.700 | 0.036 |
| 0.95 | 485.6 | 443.7 | 423.8 | 0.700 | 0.000 |
| 0.99 | 485.6 | 443.7 | 423.8 | 0.700 | 0.000 |

Interpretation: the apply gate is wired correctly and can abstain back to the frozen PPO baseline, but this small gated dataset still does not improve held-out N=4. The next useful residual move is not looser gating; it is more/better positive recovery labels, likely via broader hard-seed mining, motif-ranked failure windows, or a learned apply gate trained on counterfactual harm/benefit rather than just non-noop labels.

### Motif-Mined Gated Residual Probe

Run: `reports/n4_gated_counterfactual_residual_motifmine_20260613_seed8940k`

This expanded the gated long-option residual pass to 20 hard/near-miss collection seeds and used the learned adjacent-subchain motif prototype (`reports/n4_subchain_motif_diagnostic_20260612_seed2420k/prototype_model.json`) plus recovery pressure to rank candidate failure windows. The goal was to test whether motif-ranked windows produce denser useful residual labels while preserving solved-seed behavior.

| item | value |
|---|---:|
| rows | 160 |
| non-noop labels | 8 |
| apply labels after oversampling | 32 positive / 152 negative |
| action train accuracy | 0.875 |
| non-noop recall | 0.750 |
| apply accuracy | 0.957 |
| max score gap | 0.965 |

Held-out eval used starts `1500000` and `1600000` with 30 episodes per block.

| evaluator | mean | p10 | cvar | success | mean abs residual delta |
|---|---:|---:|---:|---:|---:|
| frozen PPO ReCoN base | 486.8 | 443.7 | 425.3 | 0.7167 | 0.000 |
| motif-mined gated residual, apply threshold 0.60 | 486.7 | 443.5 | 424.2 | 0.7167 | 0.395 |

Apply-threshold sweep: `reports/n4_gated_counterfactual_residual_motifmine_20260613_seed8940k/apply_gate_sweep.json`

| apply threshold | mean | p10 | cvar | success | mean abs residual delta |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 486.7 | 443.5 | 424.3 | 0.7167 | 0.524 |
| 0.60 | 486.7 | 443.5 | 424.2 | 0.7167 | 0.395 |
| 0.70 | 486.8 | 443.5 | 424.7 | 0.7167 | 0.298 |
| 0.80 | 486.7 | 443.5 | 424.5 | 0.7167 | 0.224 |
| 0.90 | 486.7 | 443.5 | 424.2 | 0.7167 | 0.149 |
| 0.95 | 486.8 | 443.6 | 424.8 | 0.7167 | 0.094 |

Interpretation: motif ranking improved positive-label density versus the smaller gated probe, and the apply gate prevented success degradation, but no threshold improved held-out success over the frozen PPO base. This weakens the post-hoc residual path for the current 5-bin terminal: the remaining failures are not being rescued by simple residual shifts even when counterfactual labels exist. The next better-supported direction is recurrent/on-policy curriculum or a primary policy update that uses subchain/motif state directly, rather than another residual-only pass.

### Balanced-Tail minGRU PPO KL Probe

Run: `reports/n4_mingru_ppo_balanced_tail_kl_20260613_seed8950k`

This tested a conservative PPO-style recurrent update from the best balanced DAgger7 minGRU checkpoint (`reports/n4_mingru_curriculum_subchain_motif_balanced_dagger7_20260613_seed5200k/supervised_mingru/mingru_terminal.pt`). Training used 96 seeds from `reports/n4_mingru_dagger5_hardseed_mine_20260613_seed5200k/balanced_tail_seeds.txt`, 6 rollout/update iterations of 16 episodes, LR `3e-6`, clip `0.04`, reference KL coefficient `0.08`, late-survival bonus `0.02`, and the motif/subchain/previous-force observation mode.

Rollout training stats were noisy rather than consistently improving: per-iteration success was `0.50`, `0.4375`, `0.50`, `0.5625`, `0.4375`, `0.50`, while reference KL stayed tiny (`~1e-6`). Held-out eval used the standard recurrent mixed block: starts `1900000`, `2000000`, `2100000`, `2200000`, 20 episodes each.

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| DAgger7 baseline, pure/ReCoN | 486.8 | 444.9 | 0.675 | 80 |
| PPO balanced-tail KL, pure minGRU | 486.7 | 444.9 | 0.675 | 80 |
| PPO balanced-tail KL, ReCoN-routed minGRU | 486.4 | 442.4 | 0.6625 | 80 |

Interpretation: stronger KL preservation avoided large policy drift, but did not improve the recurrent held-out plateau. As with prior minGRU PPO attempts, the pure policy remains flat and the ReCoN wrapper can dip. This reinforces the current direction: small updates to the same global recurrent terminal are unlikely to crack N=4; future learning should either change the architecture toward explicit local/subchain option composition or use a substantially different PPO setup for the primary 5-bin policy rather than another recurrent micro-update.

## Long-Option Shared Subchain Probe - 2026-06-13

Upgraded `scripts/train_subchain_pair_terminal.py` so counterfactual shared-subchain labels can score two-phase local options instead of only a single short forced action. Each candidate now has an initial force phase plus optional tail force phase. The baseline comparison is a true base-force sequence, and successful counterfactual options can add sampled forced-trace states back into the pair-terminal dataset. This lets the shared adjacent-pair terminal learn short local force patterns over time rather than only one isolated initial force.

Verification:

- `uv run ruff check scripts/train_subchain_pair_terminal.py tests/test_policy_terminal_training.py` -> passed.
- `uv run pytest tests/test_policy_terminal_training.py::test_subchain_pair_counterfactual_expands_tail_options -q -s` -> 1 passed.

Smoke: `reports/smoke_subchain_pair_optiontrace_20260613`

The earlier long-option smoke with only initial-state labels still produced no recoveries. After switching to true base-sequence comparison plus option-trace rows, the same weak-block smoke produced actionable labels: `counterfactual_recovery: 9`, `counterfactual_recovery_option: 72`, `counterfactual_no_better: 15`, and `preserve_success: 24` pair rows. Tiny held-out eval remained neutral at success `0.50` for both base and learned, so this is wiring/label evidence only.

Bounded probe: `reports/n4_subchain_pair_optiontrace_probe_20260613_seed8960k`

Setup: frozen survival PPO teacher (`reports/n4_survival_ppo_sweep_20260612_seed2700k/candidate_01/checkpoint_010000.zip`), collection seeds from the weak `980000` block, two-phase options of `20 + 20` ticks, option trace stride `4`, hidden size `48`, conservative subchain blend `0.12`, and held-out eval on starts `1500000` and `1600000` with 20 episodes per block.

Dataset source counts: `counterfactual_recovery: 27`, `counterfactual_recovery_option: 216`, `counterfactual_no_better: 78`, `preserve_success: 117`, total pair rows `438`.

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| frozen PPO ReCoN base | 485.6 | 443.7 | 0.700 | 40 |
| base + option-trace shared subchain, blend 0.12 | 485.6 | 443.7 | 0.700 | 40 |

Blend sweep on the same checkpoint and held-out seeds:

| blend | mean | p10 | success |
|---:|---:|---:|---:|
| 0.05 | 485.6 | 443.7 | 0.700 |
| 0.12 | 485.6 | 443.7 | 0.700 |
| 0.25 | 483.2 | 443.6 | 0.675 |
| 0.40 | 480.3 | 430.0 | 0.675 |
| 0.60 | 480.3 | 429.1 | 0.675 |
| 0.85 | 465.2 | 394.1 | 0.450 |
| 1.00 | 465.2 | 394.1 | 0.450 |

Interpretation: the improved labeler now finds nontrivial local option recoveries, but the learned shared-subchain terminal does not improve held-out N=4 at conservative authority and degrades once given enough blend to matter. This is useful infrastructure and negative evidence. The next performance move should not be higher subchain blend; it should either use the option-trace labels as auxiliary data for the primary recurrent/policy terminal, or return to a broader primary PPO/curriculum update. No N=4 solve claim is justified.

## Sparse PPO Continuation And Hard-Tail Follow-Up - 2026-06-13

Ran a bounded sparse continuation sweep from the current 5-bin survival PPO terminal (`reports/n4_survival_ppo_sweep_20260612_seed2700k/candidate_01/checkpoint_010000.zip`) using mixed-grid validation and held-out final blocks. This covered the requested PPO axes in a sparse way: learning rate, clip range, `n_steps`, `n_epochs`, GAE lambda, entropy, reported net arch rows, VecNormalize on/off, and late-survival bonus. Because this is checkpoint continuation, changing `net_arch` is recorded in config but SB3 restores the saved checkpoint architecture; architecture sweeps are only meaningful from scratch.

Run: `reports/n4_ppo_sparse_mixedgrid_20260613_seed8970k`

Setup: four sparse candidates, two `5000`-step chunks each, validation starts `900000`, `930000`, `970000`, `1010000`, `1040000`, `1070000`, `1140000`, and `1300000` with 8 episodes per start; final starts `1500000`, `1600000`, and `1700000` with 20 episodes per start.

| idx | grid | lr | clip | steps | epochs | gae | ent | net row | VecNorm | late bonus | final mean | final p10 | final cvar | final success |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 0 | 0 | 2.5e-7 | 0.002 | 512 | 1 | 0.95 | 0.0 | 64,64 | false | 0.005 | 484.8 | 433.8 | 418.7 | 0.683 |
| 1 | 7 | 2.5e-7 | 0.002 | 512 | 1 | 0.95 | 0.0 | 128,128 | true | 0.020 | 484.8 | 433.8 | 418.7 | 0.683 |
| 2 | 572 | 5e-7 | 0.002 | 1024 | 2 | 0.98 | 0.001 | 256,128 | false | 0.005 | 484.8 | 433.8 | 418.7 | 0.683 |
| 3 | 767 | 5e-7 | 0.004 | 1024 | 2 | 0.98 | 0.001 | 256,128 | true | 0.020 | 484.8 | 433.8 | 418.7 | 0.683 |

All four rows chose the `checkpoint_000000_start.zip` copy as best. No 10k continuation promoted over the incumbent. The VecNormalize rows were not useful when resuming from a non-VecNormalize checkpoint; chunk-level validation dipped before falling back to the start checkpoint.

Follow-up: `reports/n4_hardtail_no_rebuild_continue_20260613_seed8990k`

The first hard-tail follow-up (`reports/n4_hardtail_anchor_continue_20260613_seed8980k`) was interrupted after chunk 1 because it stalled while rebuilding a `SubprocVecEnv`; the completed chunk exactly matched the start validation and did not improve. The no-rebuild follow-up used the mined tail-seed pool from the sparse sweep, high hard-seed probability (`0.90`), LR `2e-6`, clip `0.01`, GAE `0.98`, entropy `0.001`, late-survival bonus `0.02`, and a light teacher anchor. Validation used starts `1500000`, `1600000`, and `1700000` with 15 episodes per start; final eval used starts `1900000`, `2000000`, and `2100000` with 15 episodes per start.

| checkpoint | mean | p10 | cvar | success | promoted |
|---|---:|---:|---:|---:|---|
| start | 485.6 | 442.6 | 426.4 | 0.711 | true |
| chunk_1 | 485.5 | 442.6 | 426.4 | 0.711 | false |
| chunk_2 | 482.2 | 435.6 | 403.8 | 0.689 | false |

Final held-out on `1900000`, `2000000`, `2100000` used the start checkpoint because no chunk promoted: pure PPO success `0.444`, ReCoN-routed policy terminal success `0.578` over 45 episodes. This is not a solve claim and shows that this final block is a tougher tail slice for the incumbent.

Interpretation: small continuation updates do not move the current 5-bin policy, and stronger hard-tail replay with a light teacher anchor still degrades before it helps. The next PPO attempt should avoid VecNormalize when resuming this checkpoint and should either run a genuinely from-scratch architecture sweep or change the objective/data path, for example auxiliary training on option-trace labels or a recurrent curriculum that treats hard-tail states as on-policy distribution shift rather than repeated PPO micro-updates.

## Option-Trace Auxiliary Recurrent Dataset - 2026-06-13

The long-option shared-subchain labeler now exports a second sidecar dataset, `option_policy_dataset.npz`, in the same primary-policy format consumed by `train_mingru_supervised.py`: observations, previous force, teacher force/action labels, returns-to-go, failure flags, source tags, episode ids, step indices, and sample weights. This keeps the existing pair-terminal dataset intact while making counterfactual option traces usable by the primary recurrent policy/minGRU terminal.

Implementation notes:

- `scripts/train_subchain_pair_terminal.py` records `prev_force` in teacher rollouts and forced option traces.
- Successful counterfactual options now add full-policy sidecar rows for both the initial recovery state and sampled forced option-trace states.
- `--option-policy-observation-mode` controls sidecar observations; the recurrent-compatible probe used `normalized_raw4_subchains_prev_force`.
- Focused verification: `uv run ruff check scripts/train_subchain_pair_terminal.py tests/test_policy_terminal_training.py` passed, and `uv run pytest tests/test_policy_terminal_training.py::test_subchain_pair_counterfactual_expands_tail_options tests/test_policy_terminal_training.py::test_subchain_option_trace_exports_primary_policy_rows -q -s` passed.

Dataset probe: `reports/n4_option_policy_subchain_dataset_20260613_seed9010k`

The sidecar contains 146 primary-policy rows with observation shape `(146, 23)`: `counterfactual_recovery: 9`, `counterfactual_recovery_option: 72`, `counterfactual_no_better: 26`, and `preserve_success: 39`. Action labels cover all 5 bins: `{0: 52, 1: 27, 2: 15, 3: 3, 4: 49}`.

Auxiliary finetune: `reports/n4_mingru_optiontrace_aux_20260613_seed9020k`

This resumed from the balanced DAgger7 minGRU checkpoint and trained conservatively on only the option-trace sidecar. Held-out eval used starts `1900000`, `2000000`, `2100000`, and `2200000` with 20 episodes per start.

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| DAgger7 pure minGRU | 486.8 | 444.9 | 0.675 | 80 |
| DAgger7 ReCoN-routed minGRU | 486.4 | 442.4 | 0.6625 | 80 |
| option-trace auxiliary pure minGRU | 486.4 | 446.1 | 0.675 | 80 |
| option-trace auxiliary ReCoN-routed minGRU | 486.7 | 442.5 | 0.675 | 80 |

Interpretation: this does not solve N=4, but it is the first useful sign that long-option counterfactual traces are better used as auxiliary primary-policy/recurrent data than as direct subchain actuator authority. The improvement is small: ReCoN-routed minGRU recovers from `0.6625` to `0.675` success on this block, and pure minGRU p10 improves slightly. The next stronger experiment should mix these option-trace rows with the full DAgger curriculum dataset instead of finetuning on the tiny sidecar alone.
