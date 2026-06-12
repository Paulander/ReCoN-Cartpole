# N=4 Lexicographic Microfit Update - 2026-06-12

## Status

N=4 is **still not solved**. I added success-first checkpoint promotion and used it to preserve checkpoints that reach the nominal `0.70` validation-success line, then checked those checkpoints on separate held-out seed blocks.

## Code Change

`train_policy_terminal_tail_curriculum.py` now supports:

- `--promotion-mode score` (old behavior)
- `--promotion-mode lexicographic_success`, which promotes by `(success_rate, p10_survival, cvar_survival, mean_survival, score)` after regression gates
- `--max-cvar-regression`
- `--final-seed-starts`, so final eval can cover multiple held-out seed blocks

Test result after the change: `56 passed`.

## Main Run

Run: `reports/n4_lexicographic_microfit_20260612_seed2620k`

Config summary:

- start model: `reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`
- LR `5e-7`, clip `0.01`, 5k chunks, one PPO epoch
- mixed validation: starts `900k, 930k, 970k, 1010k, 1040k, 1070k, 1140k, 1300k`, 30 episodes each
- final held-out: starts `1.5M, 1.6M`, 120 episodes each

Validation table:

| checkpoint | mean | p10 | CVaR | success | promoted |
|---|---:|---:|---:|---:|---|
| start | 484.9 | 434.9 | 414.8 | 0.696 | yes |
| 5k | 484.9 | 434.9 | 414.8 | 0.700 | yes |
| 10k | 484.9 | 434.9 | 414.8 | 0.700 | yes |
| 15k | 484.9 | 434.9 | 414.8 | 0.700 | yes |
| 20k | 484.7 | 434.9 | 412.3 | 0.700 | no, CVaR gate |
| 25k | 481.3 | 429.3 | 393.5 | 0.675 | no |
| 30k | 473.8 | 402.9 | 379.6 | 0.588 | no |

Best checkpoint by success-first promotion: `checkpoint_015000.zip`.

Final held-out eval for that checkpoint:

| eval | mean | p10 | CVaR | success | episodes |
|---|---:|---:|---:|---:|---:|
| ReCoN policy terminal | 484.5 | 441.9 | 413.2 | 0.692 | 240 |

Same held-out comparison against the original best:

| model | mean | p10 | CVaR | success | episodes |
|---|---:|---:|---:|---:|---:|
| original best | 484.5 | 441.9 | 413.2 | 0.692 | 240 |
| lexicographic 15k | 484.5 | 441.9 | 413.2 | 0.692 | 240 |

## What Changed On Validation

The 0.700 validation bump was tiny. Comparing the validation start checkpoint to the 15k checkpoint changed only six seeds:

- `930013`: 488 -> 500, failure -> success
- `930020`: 460 -> 459
- `930021`: 489 -> 487
- `1010024`: 432 -> 433
- `1070000`: 469 -> 470
- `1140004`: 489 -> 488

Net success gain on validation: exactly one seed. That did not generalize to the held-out blocks.

## Follow-Up Variant

Run: `reports/n4_lexicographic_microfit_20260612_seed2630k_hard045`

Higher hard-seed pressure (`0.45`) did not find a success-lift candidate:

| checkpoint | mean | p10 | CVaR | success | promoted |
|---|---:|---:|---:|---:|---|
| start | 484.9 | 434.9 | 414.8 | 0.696 | yes |
| 5k | 484.8 | 434.9 | 414.8 | 0.696 | yes/tie |
| 10k | 484.8 | 434.0 | 414.8 | 0.696 | no |
| 15k | 484.6 | 434.0 | 414.4 | 0.696 | no |

The run was stopped before final eval.

## Interpretation

Success-first promotion is useful and less misleading than the old scalar-only promotion near the threshold, but the current PPO continuation path mostly creates single-seed flips. The frontier appears to be a brittle local optimum: tiny updates can flip one validation seed without producing a robust held-out improvement.

Best next work should focus on a different source of signal rather than more blind continuation:

1. Build a failure-mode dataset from near-miss seeds and train a gate/specialist that is explicitly evaluated out-of-block.
2. Use distillation from the feedforward best when training recurrent policies on current N=4, so recurrence does not start 0.13 success below the feedforward frontier.
3. Add per-seed action-difference diagnostics for candidate checkpoints to identify whether updates are changing decisions only at the failure boundary or broadly damaging tail behavior.
