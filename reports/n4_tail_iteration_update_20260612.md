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
