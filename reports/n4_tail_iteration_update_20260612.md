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
