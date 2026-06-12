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
