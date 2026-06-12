# N4 Tail Iteration Update - 2026-06-12

## What changed

- Added `scripts/train_counterfactual_action_gate.py` for a learned ReCoN action-override gate trained from near-failure counterfactual probes.
- Fixed `scripts/audit_failure_actions.py` so audits use the same `policy_terminal_scope` and `policy_observation_mode` as the evaluated checkpoint. The earlier default `env` observation mode was not comparable to the current `normalized_raw` best model.
- Added diagnostic reporting for label counts, max survival gap, and max score gap so a run cannot silently present a no-op gate as useful learning.

## Corrected failure-action audit

Run: `reports/n4_failure_action_audit_normraw_20260612`

- Controller: current robust N=4 checkpoint, `hard_select`, `stabilize_chain`, `normalized_raw`.
- Audited states: 77 near-failure states from seeds starting at 980000.
- Mistake rate by tiny margin score: 0.714.
- Mean survival gap: 0.000, p90 survival gap: 0.000.
- Mean score gap: 1.04e-5, p90 score gap: 1.94e-5.

Interpretation: the current failures are not clean one-step action mistakes under a 120-step probe. Alternative actions can have microscopically better terminal margin, but they do not extend survival in this audit.

## Counterfactual margin gate

Run: `reports/n4_counterfactual_gate_20260612_seed2683k_margin_lowgap`

- Training rows: 240.
- Positive labels: 96.
- Max survival gap: 0.000.
- Max score gap: 9.49e-5.
- Held-out evaluation on seed blocks 1500000 and 1600000, 60 episodes each:
  - Base ReCoN: mean 483.5, p10 441.9, CVaR 411.8, success 0.692.
  - Gated ReCoN at confidence 0.55: mean 483.4, p10 441.8, CVaR 411.6, success 0.692, 3758 overrides.
- Conservative confidence sweep:
  - 0.6 and 0.7 slightly regressed lower-tail metrics.
  - 0.8 matched base while still overriding 440 times.
  - 0.9+ produced no overrides and exactly matched base.

Interpretation: a pointwise learned action gate can imitate tiny margin preferences but does not improve held-out N=4 survival. This branch should not be treated as progress toward solve.

## Upright-tail curriculum probe

Run: `reports/n4_upright_tail_probe_20260612_seed2691k`

- Started from `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`.
- CPU/subproc was much faster than the aborted CUDA MLP PPO attempt; CUDA stalled before the first validation chunk.
- Two 5k chunks, reward mode `upright_shaping`, mild failure penalty and late survival bonus.
- Validation slice:
  - Start: success 0.675.
  - Chunk 1: success 0.683, promoted.
  - Chunk 2: success 0.683, rejected due tail regression.
- Final held-out ReCoN eval: mean 483.3, p10 441.9, CVaR 411.8, success 0.692 over 120 episodes.
- Pure PPO checkpoint alone: success 0.600 on the same final held-out seeds.

Interpretation: mild upright shaping nudged the small validation slice but did not transfer to held-out success. ReCoN wrapper remains materially better than pure PPO alone, but this is not a new N=4 solve.

## Current state

The best robust checkpoint is still `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`, with held-out success around 0.69 on the standard mixed blocks. The evidence from this iteration says the remaining N=4 tail is not fixed by single-step action overrides; the next likely path is longer-horizon curriculum/recurrent policy work or broader PPO objective/domain randomization, evaluated only on held-out blocks.


## Compact survival PPO sweep

Run: `reports/n4_survival_ppo_sweep_20260612_seed2700k`

- Four CPU/subproc candidates from the current robust checkpoint.
- Survival reward, strict lexicographic promotion, varied LR/clip/entropy/late-survival bonus.
- Best candidate by final score: LR 5e-7, clip 0.003, entropy 0.001, late-survival bonus 0.005.
- Held-out seed blocks 1500000 and 1600000, 60 episodes each:
  - All four candidates stayed at success 0.692.
  - Mean/p10/CVaR remained effectively unchanged around mean 483.5, p10 441.9, CVaR 411.8.

Interpretation: compact PPO hyperparameter nudges are not moving the held-out N=4 tail, even when validation chunks show small promoted improvements.

## High-block validation curriculum

Run: `reports/n4_highblock_tail_curriculum_20260612_seed2710k`

- Validation blocks: 1500000, 1600000, 1700000, 1800000.
- Final held-out blocks: 1900000 and 2000000.
- The starting robust checkpoint scored 0.700 on the high-block validation slice, but no trained chunk beat it under the promotion gates.
- Final held-out ReCoN eval: mean 484.4, p10 438.0, CVaR 412.6, success 0.683 over 240 episodes.
- Pure PPO alone on the same final blocks: success 0.508.

Interpretation: 0.700 can appear on a validation slice, but it does not yet generalize to fresh held-out blocks. No N=4 solve claim is justified.


## Checkpoint model selection on fresh held-out blocks

Run: `reports/n4_model_selection_1900_2000_20260612`

Compared recent candidate checkpoints on held-out seed blocks 1900000 and 2000000, 120 episodes each:

| checkpoint | mean | p10 | cvar | success |
|---|---:|---:|---:|---:|
| base_1520k_025k | 484.4 | 438.0 | 412.6 | 0.683 |
| targetkl_seed2650k_005k | 484.4 | 438.0 | 412.6 | 0.683 |
| upright_seed2691k_005k | 484.3 | 438.0 | 412.6 | 0.671 |
| sweep2700_candidate01_010k | 484.4 | 438.0 | 412.6 | 0.683 |
| sweep2700_candidate03_010k | 484.4 | 438.0 | 412.6 | 0.683 |

Interpretation: recent promoted checkpoints are mostly action-equivalent to the base under the ReCoN wrapper on these held-out seeds; upright shaping is worse. Model selection does not reveal a hidden better checkpoint.


## Recurrent multi-block evaluation patch and bounded run

Code change: recurrent tail/curriculum scripts now support `--final-seed-starts`, matching the multi-block held-out discipline used by feedforward tail runs. Smoke run `reports/smoke_recurrent_multiblock_final_20260612` verified two final starts produced four final eval episodes.

Run: `reports/n4_recurrent_multiblock_tail_20260612_seed2720k`

- RecurrentPPO, `normalized_raw4_prev_force`, 2 x 10k chunks, validation blocks 1500000/1600000/1700000/1800000.
- Best validation checkpoint: chunk 1, success 0.483.
- Final held-out blocks: 1900000 and 2000000, 60 episodes each.
- Pure recurrent PPO final: mean 461.6, p10 383.9, CVaR 362.9, success 0.442.
- ReCoN recurrent terminal final: mean 465.0, p10 389.5, CVaR 365.7, success 0.492.

Interpretation: direct recurrent PPO from scratch is currently much weaker than the best feedforward ReCoN terminal on the same held-out block. The recurrent path likely needs distillation/warm-start from the feedforward policy or a different recurrent terminal training objective before it is competitive.


## Teacher-distilled minGRU warm-start attempt

Code changes:

- minGRU dataset/training/ladder scripts now accept padded policy observation modes: `normalized_raw4`, `normalized_raw_prev_force`, and `normalized_raw4_prev_force`.
- minGRU previous-force handling now avoids double-counting the force column when the observation mode already includes it.
- `build_policy_dataset.py` now separates student `--observation-mode` from `--teacher-observation-mode`, so a padded recurrent student can imitate the existing feedforward teacher without changing the teacher checkpoint input shape.
- `train_recurrent_terminal_ladder.py` now supports multi-block `--validation-seed-starts`.

Run: `reports/n4_mingru_distill_20260612_seed2730k`

- Teacher: current best feedforward ReCoN policy terminal, checkpoint `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`.
- Student observations: `normalized_raw4_prev_force`.
- Teacher observations: `normalized_raw`.
- Dataset: 60 episodes from seed start 1720000, 28,999 samples.
- Supervised minGRU: hidden 64, sequence length 16, 8 epochs.
- Final validation action accuracy: 0.751.
- Held-out ladder eval: seed starts 1900000 and 2000000, 60 episodes each.

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| pure_mingru_policy | 462.0 | 377.9 | 0.467 | 120 |
| recon_mingru_terminal | 463.4 | 382.9 | 0.492 | 120 |

Interpretation: distillation/warm-start made the minGRU path valid and measurable with padded previous-force inputs, but this small supervised student is still far below the feedforward ReCoN terminal. The recurrent path needs either much stronger imitation data/training or a residual recurrent objective rather than replacing the feedforward terminal.

## No-context minGRU and one-pass DAgger distillation

Code change: `build_policy_dataset.py` now supports a DAgger-style rollout policy. The labels still come from the stronger feedforward ReCoN teacher, but the environment trajectory can be generated by a minGRU student, so recurrent training can include states caused by its own closed-loop behavior.

No-context supervised sweep from the original teacher-rollout dataset:

| candidate | dataset | train acc | val acc | ReCoN mean | ReCoN p10 | ReCoN success | episodes |
|---|---|---:|---:|---:|---:|---:|---:|
| h128 seq32 no-context | teacher rollout, 28,999 samples | 0.867 | 0.874 | 471.8 | 403.7 | 0.517 | 120 |
| h256 seq32 no-context | teacher rollout, 28,999 samples | 0.903 | 0.910 | 468.6 | 395.9 | 0.525 | 120 |

DAgger-style one-pass run: `reports/n4_mingru_dagger_20260612_seed2740k`

- Behavior policy for new collection: h128 seq32 no-context minGRU from `reports/n4_mingru_distill_20260612_seed2730k/supervised_h128_seq32_noctx/mingru_terminal.pt`.
- Teacher labels: current best feedforward ReCoN terminal, `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`.
- Student-rollout dataset: 40 episodes from seed start 1740000, 19,079 samples.
- Combined training dataset: original 28,999 teacher-rollout samples + 19,079 student-rollout samples = 48,078 samples.
- Student: h128 seq32 no-context, 20 supervised epochs.
- Final supervised validation action accuracy: 0.875.
- Held-out eval: seed starts 1900000 and 2000000, 60 episodes each.

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| pure_mingru_policy | 470.7 | 376.8 | 0.650 | 120 |
| recon_mingru_terminal | 483.1 | 437.8 | 0.683 | 120 |

Interpretation: the first DAgger-style aggregation is the strongest recurrent result so far. ReCoN+minGRU now roughly matches the feedforward ReCoN terminal on this 1900k/2000k held-out slice, but it still does not exceed the best feedforward result or solve N=4. This is real learned recurrent behavior rather than a new hand-coded script node, and the next credible step is iterative DAgger/residual recurrent training, not another direct from-scratch recurrent PPO run.

## Iterative DAgger minGRU follow-up

Run root: `reports/n4_mingru_dagger_iter2_20260612_seed2760k`

Second DAgger pass:

- Behavior policy: first-pass h128 no-context minGRU.
- New behavior rollout data: 60 episodes from seed start 1760000, 28,433 samples.
- Aggregated dataset: original teacher rollout + two student-rollout sets = 76,511 samples.
- Held-out eval blocks: 1900000, 2000000, 2100000, 2200000, 60 episodes each.

| candidate | train acc | val acc | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|---:|---:|
| iter2 h128 pure minGRU | 0.864 | 0.862 | 486.0 | 443.9 | 0.708 | 240 |
| iter2 h256 pure minGRU | 0.895 | 0.895 | 486.8 | 444.0 | 0.713 | 240 |

Block detail for the best learned policy, iter2 h256:

| seed start | mean | p10 | success | episodes |
|---:|---:|---:|---:|---:|
| 1900000 | 486.8 | 445.7 | 0.700 | 60 |
| 2000000 | 485.9 | 444.9 | 0.717 | 60 |
| 2100000 | 483.8 | 439.9 | 0.633 | 60 |
| 2200000 | 490.9 | 461.8 | 0.800 | 60 |

ReCoN-wrapper check on the same iter2 h128 checkpoint still trailed the pure learned terminal: ReCoN + minGRU stayed at success 0.683 over 120 episodes, while pure minGRU reached 0.700. Changing minGRU scope from `stabilize_chain` to `all` did not change the wrapper result, so the gap is not a simple scope setting; it is likely proposal arbitration/confidence interaction or fallback behavior.

Third DAgger pass:

- Behavior policy: iter2 h256 no-context minGRU.
- New behavior rollout data: 80 episodes from seed start 1780000, 38,971 samples.
- Aggregated dataset: original teacher rollout + three student-rollout sets = 115,482 samples.
- Third-pass h256 validation action accuracy fell to 0.871.
- Broad held-out eval: mean 485.8, p10 445.0, success 0.708 over the same 240 episodes.

Interpretation: iterative DAgger is now the strongest learned-control path, and h256 iter2 is the current best learned N=4 artifact. It beats the prior feedforward ReCoN baseline on broad 4-block pure-policy success in this comparison, but it is not a solve and the 2100000 block remains the weak tail. A naive third aggregation pass did not help, suggesting the next step should be targeted hard-tail data selection or a ReCoN integration change that lets a high-confidence learned terminal control directly when it is empirically stronger than arbitration.

## ReCoN minGRU passthrough integration

Code change: `MinGRUTerminalConfig` now has an explicit, default-off high-confidence passthrough path: `passthrough_enabled` and `passthrough_confidence_floor`. ReCoN still runs proposal generation and arbitration, but when passthrough is enabled a valid minGRU terminal prediction can replace the final force after arbitration. The trace records `mingru_passthrough`, including whether it applied and the base proposal it overrode.

Motivation: the learned h256 minGRU policy from DAgger iteration 2 was stronger than the ReCoN-wrapped minGRU proposal path. The previous wrapper averaged/arbitrated away some of the learned policy's advantage.

Evaluation: best learned checkpoint `reports/n4_mingru_dagger_iter2_20260612_seed2760k/supervised_h256_seq32_noctx/mingru_terminal.pt`, `passthrough_enabled=True`, `passthrough_confidence_floor=0.05`.

| mode | seed blocks | mean | p10 | success | episodes |
|---|---|---:|---:|---:|---:|
| ReCoN minGRU passthrough | 1900000, 2000000 | 486.3 | 444.9 | 0.708 | 120 |
| ReCoN minGRU passthrough | 1900000, 2000000, 2100000, 2200000 | 486.8 | 444.0 | 0.713 | 240 |

The 4-block passthrough result exactly matches the pure h256 minGRU broad eval, so the ReCoN integration no longer degrades the learned recurrent controller. This is still not an N=4 solve: the 2100000 block remains the weak tail at 0.633 success.

## Targeted hard-tail DAgger attempt

Code change: `build_policy_dataset.py` now accepts `--seed-list`, allowing targeted DAgger collection on explicit hard seeds instead of only contiguous seed ranges.

Run root: `reports/n4_mingru_hardtail_20260612_seed2300k`

- Hard-seed scan policy: current best h256 minGRU from DAgger iteration 2.
- Scan pool: 240 non-eval training seeds from seed start 2300000.
- Scan result: mean 488.3, p10 447.7, success 0.733.
- Selected hard/near-hard seeds: 64 seeds, chosen from failures or episodes with survival <= 475.
- Targeted DAgger samples: 29,192 student-rollout samples on those explicit hard seeds, labeled by the feedforward ReCoN teacher.
- Training dataset: iter2 aggregate 76,511 samples + hard-tail 29,192 samples = 105,703 samples.
- Student: h256 seq32 no-context minGRU, 24 epochs, validation action accuracy 0.884.

Held-out eval with ReCoN minGRU passthrough on seed starts 1900000, 2000000, 2100000, 2200000, 60 episodes each:

| checkpoint | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| best iter2 h256 passthrough | 486.8 | 444.0 | 0.713 | 240 |
| targeted hard-tail h256 passthrough | 485.1 | 438.9 | 0.696 | 240 |

Block detail for targeted hard-tail h256:

| seed start | mean | p10 | success | episodes |
|---:|---:|---:|---:|---:|
| 1900000 | 484.8 | 437.4 | 0.667 | 60 |
| 2000000 | 484.0 | 443.9 | 0.683 | 60 |
| 2100000 | 482.3 | 435.8 | 0.633 | 60 |
| 2200000 | 489.4 | 449.9 | 0.800 | 60 |

Interpretation: naive hard-tail oversampling from a separate training pool did not improve the held-out weak block and regressed overall success. Keep iter2 h256 as the current best learned N=4 checkpoint. The next hard-tail attempt should probably avoid pure oversampling and instead use either weighted sampling/mixing, a residual correction head, or collect hard seeds that specifically match the 2100000 failure signature without training on the held-out block itself.

## Warm-start hard-tail fine-tune and passthrough confidence sweep

Code change: `train_mingru_supervised.py` now accepts `--resume-checkpoint`, loading compatible minGRU weights before supervised training while preserving the requested training config. This enables conservative fine-tuning from the current best learned policy instead of retraining from scratch.

Warm-start hard-tail run: `reports/n4_mingru_warm_hardtail_20260612_seed2310k`

- Resume checkpoint: best iter2 h256 minGRU, `reports/n4_mingru_dagger_iter2_20260612_seed2760k/supervised_h256_seq32_noctx/mingru_terminal.pt`.
- Training data: iter2 aggregate + targeted hard-tail dataset, 105,703 samples.
- Fine-tune: h256 seq32 no-context, 8 epochs, LR 5e-5, max grad norm 0.75.
- Validation action accuracy: 0.861.
- Held-out ReCoN passthrough eval on seed starts 1900000, 2000000, 2100000, 2200000: mean 484.8, p10 437.8, success 0.692 over 240 episodes.

Interpretation: warm-start hard-tail fine-tuning still regressed versus the best iter2 h256 checkpoint. This supports the hypothesis that feedforward-teacher labels on the selected hard-tail states are not a good correction target for the stronger minGRU policy.

Passthrough confidence-floor sweep on best iter2 h256:

| passthrough floor | eval blocks | mean | p10 | success | episodes |
|---:|---|---:|---:|---:|---:|
| 0.05 | 2100000 | 483.8 | 439.9 | 0.633 | 60 |
| 0.30 | 2100000 | 483.8 | 439.9 | 0.633 | 60 |
| 0.50 | 2100000 | 483.8 | 439.9 | 0.633 | 60 |
| 0.70 | 2100000 | 483.8 | 439.9 | 0.633 | 60 |
| 0.90 | 2100000 | 483.8 | 439.9 | 0.650 | 60 |
| 0.93 | 2100000 | 483.8 | 439.9 | 0.650 | 60 |
| 0.95 | 2100000 | 483.8 | 439.8 | 0.650 | 60 |
| 0.97 | 2100000 | 483.7 | 439.8 | 0.650 | 60 |

Broad held-out eval with best iter2 h256 and `passthrough_confidence_floor=0.90`:

| seed start | mean | p10 | success | episodes |
|---:|---:|---:|---:|---:|
| 1900000 | 486.9 | 446.0 | 0.700 | 60 |
| 2000000 | 485.9 | 444.9 | 0.733 | 60 |
| 2100000 | 483.8 | 439.9 | 0.650 | 60 |
| 2200000 | 490.9 | 461.8 | 0.800 | 60 |
| total | 486.9 | 444.9 | 0.721 | 240 |

Interpretation: the learned confidence head is useful only at a high threshold, but a strict passthrough floor improves the integrated ReCoN+minGRU result from 0.713 to 0.721 success over the broad held-out comparison. This is the current best N=4 result, still not a solve. The next productive path is likely learning a better gate/residual around the minGRU policy, not more feedforward-teacher hard-tail cloning.

## MinGRU gate, weighting, and capacity probes

Code changes:

- Added `passthrough_logit_margin_floor` to the minGRU terminal config. ReCoN passthrough traces now record the recurrent policy logit margin and the configured margin floor, so the viewer/report can show whether passthrough happened because the learned terminal was confident and decisive.
- Added supervised minGRU sample weighting knobs: `--failure-sample-weight`, `--late-sample-weight`, and `--low-return-sample-weight`. Defaults are zero, so existing training behavior is unchanged unless explicitly enabled.
- Added `--device` support to minGRU supervised training and the recurrent ladder; CUDA is visible in this environment as an RTX 3090.

Focused weak-block gate: all candidates below were evaluated on held-out seed start `2100000`, 60 episodes, horizon 500, strict ReCoN minGRU passthrough with `passthrough_confidence_floor=0.90`.

| candidate | mean | p10 | success | note |
|---|---:|---:|---:|---|
| current best iter2 h256 seq32 | 483.8 | 439.9 | 0.650 | incumbent weak-block result |
| margin floor 0.05/0.10/0.20/0.35 | 483.3 | 439.9-440.3 | 0.633 | filtering by logit margin removed useful saves |
| weighted warm-start h256 seq32 | 482.4 | 437.6 | 0.633 | higher imitation accuracy did not transfer to control |
| h512 seq32 from scratch | 482.4 | 436.7 | 0.633 | extra capacity did not improve tail control |
| h256 seq64 from scratch | 482.4 | 436.7 | 0.617 | longer recurrent history regressed |

Interpretation: the current N=4 gap is not fixed by post-hoc margin gating, simple tail-weighted imitation, larger hidden state, or longer sequence length. The best checkpoint remains `reports/n4_mingru_dagger_iter2_20260612_seed2760k/supervised_h256_seq32_noctx/mingru_terminal.pt` with strict passthrough floor `0.90`, broad held-out success `0.721` over seed starts 1900000, 2000000, 2100000, and 2200000. No N=4 solve claim is justified.

Next likely productive move: change the data-generation objective, not just the supervised learner. The DAgger labels still come from the feedforward teacher, and the failures are not clean one-step action mistakes. A recurrent/residual objective that directly rewards late recovery, or a curriculum that collects successful recovery trajectories from easier N=4 distributions before current-noise N=4, is more plausible than more uniform teacher cloning.

## Success-filtered minGRU cloning probe

Code change: minGRU supervised training now supports `--min-sample-episode-survival` and `--max-sample-episode-survival`. The filter estimates each sample's episode survival as `returns_to_go + step_indices`, records the retained fraction in `report.json`, and is also exposed through the recurrent ladder.

Dataset check on the iter2 DAgger dataset:

| min survival | kept samples | kept fraction |
|---:|---:|---:|
| 450 | 64,214 | 0.839 |
| 470 | 57,310 | 0.749 |
| 480 | 56,352 | 0.737 |
| 490 | 52,480 | 0.686 |
| 500 | 50,000 | 0.654 |

Run: `reports/n4_mingru_success_filter_20260612_seed2350k/supervised_h256_seq32_success500_warm`

- Warm-started from the current best iter2 h256/seq32 checkpoint.
- Trained only on full-survival samples with `min_sample_episode_survival=500`.
- Low LR `3e-5`, 4 epochs, CUDA.
- Weak held-out block `2100000`, 60 episodes, strict passthrough floor `0.90`: mean `482.5`, p10 `438.5`, success `0.633`.

Interpretation: success-filtered cloning is also dominated by the incumbent weak-block result of `0.650`. The failure mode is not solved by cloning only trajectories that already survived; the next data objective probably needs counterfactual/rewarded recovery training rather than more teacher-action filtering.

## Residual recovery shaping and ReCoN-aligned residual training

Code changes:

- `train_residual_policy_terminal.py` now supports optional potential-based recovery shaping. The residual reward can include pressure reduction from the previous tick to the next tick via `--recovery-progress-weight`, plus explicit `--failure-penalty` and `--success-bonus` terms. Defaults are zero, so prior residual runs are reproducible.
- Residual training now supports `--residual-base-controller recon_policy_terminal`, which freezes and uses the same ReCoN+PPO controller during residual training that is used during ReCoN-integrated evaluation. This avoids training a residual on pure PPO and then deploying it on a different base policy.

Pure-PPO-base residual run: `reports/n4_residual_recovery_shaping_20260612_seed2360k`

- Base: frozen feedforward PPO terminal `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`.
- Residual: bin-delta PPO, proposal diagnostics, 50k timesteps, recovery shaping enabled.
- Weak held-out block `2100000`, 60 episodes:

| evaluator | mean | p10 | cvar | success |
|---|---:|---:|---:|---:|
| residual_env_frozen_ppo | 436.7 | 321.7 | 298.0 | 0.483 |
| residual_env_specialist | 447.8 | 342.8 | 313.7 | 0.500 |
| recon_frozen_base | 482.9 | 441.2 | 414.3 | 0.633 |
| recon_residual_specialist | 482.3 | 436.7 | 413.7 | 0.633 |

Gate sweep on the learned residual (`0.30` through `0.90`) did not beat frozen ReCoN. The best rows tied success at `0.633` and approached the frozen-base p10 only when the gate suppressed most residual changes.

ReCoN-base residual run: `reports/n4_residual_reconbase_recovery_20260612_seed2370k`

- Residual trained directly on top of frozen ReCoN+PPO using `--residual-base-controller recon_policy_terminal`.
- 30k timesteps, gate threshold `0.60`, stronger low-risk change penalty.
- Weak held-out block `2100000`, 60 episodes:

| evaluator | mean | p10 | cvar | success |
|---|---:|---:|---:|---:|
| recon_frozen_base | 482.9 | 441.2 | 414.3 | 0.633 |
| reconbase_residual_env | 482.9 | 441.2 | 414.3 | 0.633 |
| recon_residual_integrated | 482.9 | 441.2 | 414.3 | 0.633 |

Gate sweep over thresholds `0.30`, `0.45`, `0.60`, `0.75` found no improvement. Threshold `0.30` regressed p10 to `439.4`; `0.45+` effectively tied the frozen base.

Interpretation: recovery-shaped residual learning is now better aligned and no longer obviously harmful, but it has not cracked the N=4 weak block. The pure-PPO-base residual learned real corrections for pure PPO, yet those corrections were redundant or harmful once ReCoN arbitration was present. The ReCoN-base residual learned to be conservative. The next residual attempt should likely use an explicit advantage-style objective over paired base-vs-residual rollouts or train on saved near-failure states with short-horizon counterfactual rollouts, rather than ordinary PPO survival reward.

## Counterfactual residual terminal probe

Code changes:

- ReCoN residual terminals can now load Torch `.pt` classifiers in addition to SB3 PPO `.zip` policies. The classifier exposes the same `predict()` shape as the PPO residual policy, so it uses the existing residual terminal path and trace fields rather than an external action override.
- Added `scripts/train_counterfactual_residual_terminal.py`. It collects near-failure states from a frozen ReCoN+PPO controller, probes residual bin shifts (`-2..+2`) for a short horizon, trains a no-op/shift classifier, then evaluates the learned residual through normal ReCoN residual-terminal integration.
- The script supports contiguous collection seeds or an explicit `--collect-seed-list` JSON/text file so it can target known hard-seed pools without using held-out evaluation blocks.

Easy contiguous collection run: `reports/n4_counterfactual_residual_20260612_seed2380k`

- Collection seeds: `2380000..2380079`.
- Dataset: 321 rows, 27 failing episodes, 0 non-noop residual labels.
- Held-out weak block `2100000`, 60 episodes: base success `0.633`, residual success `0.633`.

Hard-pool collection run: `reports/n4_counterfactual_residual_hardpool_20260612_seed2381k`

- Collection seeds: first 80 seeds from `reports/n4_targetkl_survival_tail_20260612_seed2650k/tail_seed_pool.json`.
- Dataset: 411 rows, 57 failing episodes, 1 non-noop residual label.
- Held-out weak block `2100000`, 60 episodes: base success `0.633`, residual success `0.633`.

Interpretation: the counterfactual residual infrastructure works, but one-step residual bin shifts almost never produce a clear short-horizon advantage under the current ReCoN+PPO base. This supports the earlier audit result: remaining N=4 failures are not usually clean one-tick action mistakes. A useful residual probably needs multi-step/options-style correction, a value/advantage target over trajectories, or a recurrent terminal trained with an objective that directly optimizes recovery instead of labeling isolated first actions.

## Residual option-hold, self-imitation, and subchain representation probe

Code changes:

- Residual bin-delta terminals now support `residual_policy_terminal_hold_steps`, allowing a nonzero learned residual shift to persist as a short option across ticks. Traces record requested/applied shift, hold window, reuse, and remaining option ticks.
- `train_counterfactual_residual_terminal.py` can probe and train these multi-tick residual options with `--option-hold-steps`.
- `build_policy_dataset.py` now separates `--label-source teacher` from `--label-source rollout`, enabling explicit self-imitation datasets from the actual rollout policy rather than silently calling them teacher DAgger.
- `train_recurrent_terminal_ladder.py` now exposes minGRU passthrough gates (`--passthrough-enabled`, `--passthrough-confidence-floor`, `--passthrough-logit-margin-floor`) so ladder evaluations can reproduce the current best integrated ReCoN path.
- Added `normalized_raw4_subchains` and `normalized_raw4_subchains_prev_force` observation modes. These keep padded raw N=4 features and append adjacent-pair phase features for `(0,1)`, `(1,2)`, and `(2,3)`, which is the first concrete step toward letting ReCoN see N=4 as overlapping local subchains.
- `train_mingru_supervised.py` supports `--resume-partial-input`, copying compatible old checkpoint columns into widened input layers so new feature columns can be added without training the whole recurrent policy from scratch.

Experiments:

| run | weak block | mean | p10 | success | note |
|---|---:|---:|---:|---:|---|
| incumbent h256/seq32 strict passthrough | 2100000 | 483.8 | 439.9 | 0.650 | current weak-block bar |
| option-hold counterfactual residual, hold 4 | 2100000 | tied base | tied base | 0.633 | found 3 non-noop labels from 384 rows, no eval gain |
| self-imitation h256/seq32 success500 warm | 2100000 | 482.6 | 441.2 | 0.633 | cloning only successful learned rollouts regressed success |
| subchain-feature h256/seq32 partial-warm success500 | 2100000 | 482.1 | 435.8 | 0.617 | new local-chain features work technically but simple imitation regressed |

Interpretation: the infrastructure now supports multi-tick residual options, rollout-label self-imitation, and explicit adjacent-subchain inputs. None of these variants cracked the N=4 weak block. This strengthens the diagnosis that the next useful ReCoN move is not more cloning, but an explicit subchain prototype/gating objective: learn reusable successful/failing local phase motifs and use them to gate or bias the global minGRU/ReCoN action only when the motif evidence is strong. No N=4 solve claim is justified; the best held-out broad result remains the incumbent h256/seq32 minGRU strict passthrough at about 0.721 over the 1900000/2000000/2100000/2200000 blocks.

## Subchain motif prototype diagnostic

Code change: added `scripts/train_subchain_motif_gate.py`, a diagnostic-only learner that extracts adjacent-subchain phase motifs from traced N=4 rollouts, labels rows that are within a configurable future failure window, fits positive/negative prototypes, and evaluates held-out AUC. It can run with pure local subchain features or append ReCoN-visible controller diagnostics: force, minGRU confidence, failure probability, value, hidden norm, and passthrough logit margin. The script writes `report.json`, `prototype_model.json`, and `report.md`, and marks `control_policy_changed=false` so it cannot be mistaken for a solve attempt.

Runs used the current incumbent h256/seq32 minGRU strict passthrough checkpoint, training motifs on non-held-out seeds `2420000..2420079` and evaluating on weak held-out block `2100000..2100059`, sample stride 5, failure window 80.

| feature set | train AUC | held-out AUC | held-out positive rows | held-out success | interpretation |
|---|---:|---:|---:|---:|---|
| local subchain phase only | 0.606 | 0.570 | 336 | 0.650 | weak reusable signal, not enough for a gate |
| subchain + ReCoN/minGRU diagnostics | 0.897 | 0.835 | 336 | 0.650 | strong near-failure detector; suitable candidate for a future learned gate/rescue trigger |

Interpretation: this is the first strong evidence that reusable local N=4 structure exists when combined with the recurrent terminal's own uncertainty/value signals. The detector does not yet affect control, so no performance improvement or solve claim is made. The next control-facing experiment should use this score conservatively: trigger additional recovery data collection, adjust passthrough only under high motif-risk, or train a residual option conditioned on motif-risk instead of applying a hand-coded rescue.

## Online motif-gated passthrough control check

Code change: added `scripts/evaluate_motif_gated_passthrough.py`, which fits the subchain+ReCoN diagnostic prototype on non-held-out seeds, then uses motif-risk online during held-out evaluation. It tests reversible action policies outside the core controller: baseline, suppress minGRU passthrough when motif-risk is high, and force minGRU passthrough when motif-risk is high. This is a causal control ablation, not a train-seed solve claim.

Compact held-out weak-block run: `reports/n4_motif_gated_passthrough_20260612_weak20`, train seeds `2420000..2420079`, held-out seeds `2100000..2100019`, failure window 80, thresholds from positive motif-score percentiles.

| mode | best/representative threshold | mean | p10 | success | changed actions |
|---|---:|---:|---:|---:|---:|
| baseline | inf | 489.5 | 474.8 | 0.650 | 0 |
| suppress passthrough | 0.454 | 489.5 | 474.8 | 0.650 | 7 |
| force passthrough | 5.596 | 489.5 | 474.8 | 0.650 | 34 |
| force passthrough | 0.454-4.544 | 489.4 | 474.8 | 0.600 | 86-89 |

Interpretation: motif-risk is a strong near-failure detector, but a naive online action gate does not improve control. Forcing passthrough under motif-risk can hurt. Suppressing passthrough is mostly neutral because strict passthrough already changes few high-risk decisions. The next useful route is to use motif-risk for targeted residual-data collection or advantage labeling, not as a direct hand-authored action switch. No N=4 solve claim is justified.

## Subchain-diagnostic residual terminal probe

Code changes:

- Added `subchain_diagnostics` as a residual terminal feature mode. It extends proposal diagnostics with adjacent-pair phase features for `(0,1)`, `(1,2)`, and `(2,3)`, matching the newer subchain observation path.
- `evaluate_recon_residual_grid.py` now forwards and records `residual_hold_steps`, so option-hold residual checkpoints can be evaluated without silently reverting to one-tick residual changes.
- Residual training/evaluation CLIs now accept the same `subchain_diagnostics` feature mode, with tests covering feature size, controller wiring, and hold-step forwarding.

Run: `reports/n4_counterfactual_residual_subchain_20260612_seed2390k`

- Base: frozen feedforward PPO terminal `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`, evaluated through ReCoN.
- Collection seeds: targeted hard-tail pool from `reports/n4_targetkl_survival_tail_20260612_seed2650k/tail_seed_pool.json`.
- Residual labels: short-horizon counterfactual bin shifts with `--option-hold-steps 4`, `--probe-horizon 120`, no counterfactual force noise.
- Dataset: 250 rows, 170 non-noop labels.
- Label counts: `0:72`, `1:2`, `2:80`, `3:51`, `4:45`.
- Training accuracy: 0.608; non-noop recall: 0.706.
- Weak held-out block `2100000`, 60 episodes: base success `0.633`, residual success `0.633`.

Gate sweep: `reports/n4_counterfactual_residual_subchain_gate_sweep_20260612`

| threshold | max force | mean | p10 | cvar | success | episodes |
|---:|---:|---:|---:|---:|---:|---:|
| 0.450 | 4.00 | 482.8 | 441.2 | 413.7 | 0.633 | 60 |
| 0.600 | 4.00 | 482.8 | 441.2 | 414.0 | 0.633 | 60 |
| 0.750 | 4.00 | 482.8 | 441.2 | 414.0 | 0.633 | 60 |
| 0.900 | 4.00 | 482.9 | 441.2 | 414.2 | 0.633 | 60 |

Interpretation: subchain diagnostics make the residual label problem much less sparse than the earlier one-tick residual probes, so the representation is moving in the right direction. The learned residual still does not improve held-out control once integrated into ReCoN; it mostly learns interventions that are not useful enough under the weak-block dynamics. The next residual attempt should optimize an explicit trajectory advantage/recovery target, or use motif-risk to select multi-step recovery windows, rather than trusting permissive short-horizon bin labels. No N=4 solve claim is justified.

## Advantage-gated residual label calibration

Code changes:

- `train_counterfactual_residual_terminal.py` now supports stricter counterfactual label gates: `--min-survival-gain` and `--min-margin-gain` in addition to `--min-score-gap`.
- Residual reports now record the active label gates plus chosen/best survival and margin gain statistics, so a residual run can distinguish high-volume score labels from actual recovery-advantage labels.
- Added a unit test proving that `label_state` suppresses a high-scoring residual class unless it clears the configured survival-gain gate.

Calibration run: `reports/n4_counterfactual_residual_advantage_subchain_20260612_seed2392k_calib`

- Feature mode: `subchain_diagnostics`.
- Option hold: 6 ticks.
- Probe horizon: 180 ticks, no counterfactual force noise.
- Label gates: `min_score_gap=0.10`, `min_survival_gain=1`, `min_margin_gain=-0.02`.
- Dataset: 212 rows, 7 non-noop labels.
- Label counts: `0:0`, `1:3`, `2:205`, `3:3`, `4:1`.
- Mean chosen survival gain: 0.033; max best survival gain: 1.000.
- Weak-block calibration eval over seeds `2100000..2100039`: frozen base success `0.600`, residual success `0.600`, mean abs residual delta `0.045`.

A stricter attempt with `min_survival_gain=2` produced 194 rows and 0 non-noop labels before being terminated. This is useful negative evidence: residual actions that produce multi-tick survival advantage are extremely sparse under the current short-horizon counterfactual probe.

Interpretation: advantage-gated labels are much more causally honest than the permissive subchain labels, but they are too sparse to move held-out control in this form. The next residual path should probably optimize trajectory-level advantage/recovery directly, or train a recurrent recovery option on selected windows, rather than fitting a point classifier to isolated rare labels. No N=4 solve claim is justified.

## Explicit minGRU N3-to-N4 curriculum attempt

Code changes:

- Added `scripts/train_mingru_curriculum.py`, an explicit data-level recurrent curriculum wrapper. It collects stages in order: `n3_stable`, `n4_low_angle_no_noise`, `n4_current`, and `n4_hard_tail`, aggregates them with episode-id offsets, trains a minGRU terminal, and evaluates held-out N=4 seeds with pure minGRU and ReCoN+passthrough modes.
- `build_policy_dataset.py` now accepts JSON seed-list files with `hard_seeds`, `seeds`, or `tail_seeds`, matching the hard-seed pool format used elsewhere.
- Explicit seed-list collection is now bounded by `--episodes`; the first full curriculum launch exposed that the dataset builder otherwise attempted to consume the entire hard-tail JSON pool.
- Added tests covering script import, curriculum stage ordering, aggregate episode offsets, JSON hard-seed parsing, and explicit seed-list episode limiting.

Smoke run: `reports/smoke_mingru_curriculum_20260612`

- End-to-end tiny curriculum completed successfully and produced a minGRU checkpoint.

Main bounded run: `reports/n4_mingru_curriculum_20260612_seed2812k`

- Warm-start: incumbent h256/seq32 no-context minGRU from DAgger iteration 2.
- Dataset stages:
  - N3 stable/static ReCoN: 40 episodes, 20,000 samples.
  - N4 low angle/no noise/feedforward ReCoN teacher: 40 episodes, 20,000 samples.
  - N4 current/feedforward ReCoN teacher: 60 episodes, 29,155 samples.
  - N4 hard-tail behavior minGRU + feedforward ReCoN labels: 80 bounded hard seeds, 38,476 samples.
- Aggregate dataset: 107,631 samples.
- Training: h256 seq32, `normalized_raw4_prev_force`, no context, resumed from incumbent, 12 epochs, CUDA, weighted failure/late/low-return samples.
- Final validation action accuracy: 0.804.

Tiny held-out read on seed starts `1900000`, `2000000`, `2100000`, `2200000`, 2 episodes each:

| checkpoint | pure mean | pure p10 | pure success | ReCoN mean | ReCoN p10 | ReCoN success |
|---|---:|---:|---:|---:|---:|---:|
| incumbent h256/seq32 | 500.0 | 500.0 | 1.000 | 500.0 | 500.0 | 1.000 |
| curriculum h256/seq32 | 480.0 | 439.1 | 0.500 | 484.4 | 439.1 | 0.625 |

A full 4x60 eval was started but stopped after training because stepwise h256/seq32 minGRU evaluation was too slow through the current pure+ReCoN ladder path. A 4x20 eval was also stopped for the same reason. The tiny 4x2 comparison is only a direction check, not solve evidence.

Interpretation: the explicit N3-to-N4 curriculum infrastructure now exists and works, but this first warm-started curriculum dataset regressed relative to the incumbent on the same tiny held-out seeds. The likely issue is destructive imitation mixing: N3/static-ReCoN and low-angle teacher data diluted the incumbent N4 behavior instead of improving the hard tail. The next recurrent attempt should keep the infrastructure but change weighting/sampling: much lower N3/low-angle weight after warm-start, or use curriculum pretraining only before incumbent DAgger, not as a late fine-tune mix. No N=4 solve claim is justified.


## Weighted minGRU curriculum follow-up

Code changes:

- `scripts/train_mingru_curriculum.py` now supports per-stage `sample_weights` for `n3_stable`, `n4_low_angle_no_noise`, `n4_current`, and `n4_hard_tail` stages.
- `scripts/train_mingru_supervised.py` now multiplies dataset-level sample weights with failure/late/low-return sample weighting and records dataset weight stats in `report.json`.
- The curriculum runner can skip final eval with `--final-eval-episodes 0`, allowing faster training-only candidate generation followed by explicit held-out ladder checks.
- Tests now cover stage weight propagation, aggregate sample-weight preservation, and dataset-weight multiplication.

Run: `reports/n4_mingru_curriculum_weighted_20260612_seed2813k`

- Warm-start: incumbent h256/seq32 no-context minGRU from DAgger iteration 2.
- Same 107,631-sample N3-to-N4 curriculum as the previous attempt.
- Stage weights: N3 `0.05`, N4 low-angle `0.10`, N4 current `0.50`, N4 hard-tail `2.00`.
- Effective dataset weight mean before normalization: `0.878`; max raw dataset weight: `2.0`; max final combined sample weight: `3.68`.
- Training: h256 seq32, `normalized_raw4_prev_force`, no context, 8 epochs, CUDA, resumed from incumbent.
- Final validation action accuracy: `0.632`.

Tiny held-out read on seed starts `1900000`, `2000000`, `2100000`, `2200000`, 2 episodes each:

| checkpoint | pure mean | pure p10 | pure success | ReCoN mean | ReCoN p10 | ReCoN success |
|---|---:|---:|---:|---:|---:|---:|
| incumbent h256/seq32 | 500.0 | 500.0 | 1.000 | 500.0 | 500.0 | 1.000 |
| unweighted curriculum | 480.0 | 439.1 | 0.500 | 484.4 | 439.1 | 0.625 |
| weighted curriculum | 500.0 | 500.0 | 1.000 | 500.0 | 500.0 | 1.000 |

Weak-block held-out read for weighted curriculum on seeds `2100000..2100019`:

| evaluator | mean | p10 | success | episodes |
|---|---:|---:|---:|---:|
| pure minGRU | 488.1 | 463.7 | 0.600 | 20 |
| ReCoN minGRU passthrough | 488.6 | 464.1 | 0.600 | 20 |

Interpretation: stage weighting prevents the obvious destructive-regression seen in the unweighted curriculum, so the curriculum machinery is now safer for warm-start experiments. It still does not close the 2100000 weak-block tail; success remains around 0.60 on this 20-seed read. No N=4 solve claim is justified. The next performance move should shift from broad curriculum mixing toward targeted recovery-window learning or a stronger recurrent objective that directly optimizes weak-block trajectories rather than imitating the feedforward teacher on mixed stages.

## Exact-corner PPO sweep harness and bounded run

Code changes:

- `scripts/run_ppo_sweep.py` now preserves each candidate's original `grid_index` from the full Cartesian sweep.
- Added `--candidate-indices`, allowing exact grid points to be run in a deliberate order instead of relying only on stride/offset sampling.
- Fixed duplicate `final_seed_starts` reporting in intermediate summaries and updated markdown to show multi-block final seed starts.
- Added tests for exact-index candidate selection and grid-index preservation under strided candidate ordering.

Smoke run: `reports/smoke_ppo_exact_index_sweep_20260612`

- Verified the exact-index CLI path on N=4 with a 64-timestep candidate.
- Summary recorded `grid_index=0` and completed without a solve claim.

Bounded exact-corner run: `reports/n4_ppo_exact_corner_sweep_20260612_seed2831k`

- Start model: frozen current best PPO terminal `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`.
- Environment: current 5-bin N=4 serial-lagrange setup, `dt=0.0005`, `force_noise=0.02`, `link_coupling=12`, `normalized_raw` policy observations.
- Validation starts: `1500000`, `1600000`, `2100000`, 5 episodes each.
- Final starts: `1900000`, `2100000`, 5 episodes each.
- Candidate grid corners selected from LR/clip/n_steps/GAE/entropy/net/VecNormalize/late-survival axes: `0`, `85`, `170`, `255`.
- Each candidate used one 5k PPO chunk, CPU/subproc, hard-select ReCoN terminal evaluation.

| grid | lr | clip | n_steps | gae | ent | net | vecnorm | late bonus | final mean | final p10 | final success |
|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| 0 | 2.5e-6 | 0.015 | 512 | 0.90 | 0.000 | 64,64 | false | 0.00 | 481.5 | 432.2 | 0.500 |
| 85 | 2.5e-6 | 0.050 | 512 | 0.98 | 0.000 | 256,128 | false | 0.05 | 481.5 | 432.2 | 0.500 |
| 170 | 1.0e-5 | 0.015 | 1024 | 0.90 | 0.001 | 64,64 | true | 0.00 | 481.5 | 432.2 | 0.500 |
| 255 | 1.0e-5 | 0.050 | 1024 | 0.98 | 0.001 | 256,128 | true | 0.05 | 481.5 | 432.2 | 0.500 |

Best selected checkpoint remained `checkpoint_000000_start.zip` for every candidate, meaning no 5k PPO chunk beat the frozen start model under the promotion gates. This is not a solve attempt and the final eval is intentionally tiny, but it is useful evidence that short corner sweeps over these PPO knobs are action-equivalent or too weak to move the N=4 tail. The PPO path probably needs longer chunks, different reward/teacher anchoring, or a recovery-specific objective before it is worth spending larger compute.


## Preserve-success residual PPO slice

Code changes:

- `scripts/train_residual_policy_terminal.py` now supports `--preserve-base-success-penalty`.
- On seeded residual-training episodes, the env first rolls out the frozen base controller. If the base would solve that episode, residual force/bin changes receive an additional normalized penalty.
- Residual traces now expose `base_episode_success`, `base_episode_steps`, and `preserve_base_success_penalty`, making it clear when the learner is being discouraged from rewriting successful behavior.
- Added a unit test proving that residual changes on a base-solved episode are penalized and traced.

Smoke run: `reports/smoke_residual_preserve_success_20260612`

- Verified end-to-end residual PPO training/evaluation with the preservation penalty enabled.

Bounded weak-block run: `reports/n4_residual_preserve_success_20260612_seed2841k`

- Frozen base: current best PPO terminal inside ReCoN, `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`.
- Residual learner: PPO, 5-bin `bin_delta`, `subchain_diagnostics`, risk gate `0.45`, 5k timesteps.
- Training seeds: existing hard-tail pool from `reports/n4_targetkl_survival_tail_20260612_seed2650k/tail_seed_pool.json` with probability `0.70`.
- Reward shaping: low-risk change penalty `0.10`, preserve-base-success penalty `0.25`, late survival bonus `0.02`, recovery-progress weight `0.25`, failure penalty `1.0`, success bonus `5.0`.
- Held-out eval: weak block `2100000..2100019`, 20 episodes.

| evaluator | mean | p10 | cvar | success | mean abs residual delta | episodes |
|---|---:|---:|---:|---:|---:|---:|
| residual_env_frozen_base | 489.5 | 468.8 | 429.5 | 0.650 | 0.000 | 20 |
| residual_env_specialist | 489.4 | 467.9 | 429.5 | 0.650 | 0.752 | 20 |
| recon_frozen_base | 489.5 | 468.8 | 429.5 | 0.650 | 0.000 | 20 |
| recon_residual_specialist | 489.4 | 467.9 | 429.5 | 0.650 | 0.937 | 20 |

Interpretation: this is a more faithful learned residual setup than earlier permissive counterfactual labels: the base is frozen, residuals see risk/proposal/subchain diagnostics, and successful base behavior is explicitly protected. It still does not improve the held-out weak block after a 5k PPO slice. The residual path is not dead, but this result argues that simple residual PPO over whole episodes is too diffuse; a future residual attempt should train from selected recovery windows or optimize a trajectory-level recovery objective rather than hoping sparse late failures dominate normal rollout training. No N=4 solve claim is justified.
