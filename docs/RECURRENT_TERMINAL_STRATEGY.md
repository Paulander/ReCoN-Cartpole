# Recurrent Terminal Strategy

This is the updated project direction as of 2026-06-10. The repo now has enough learning machinery. The next goal is not to add random knobs; it is to make the learning loop faster, recurrent, fail-fast, and attribution-clean.

## Summary

After extensive N=4 policy-terminal work, the best controller is near-solved but not robustly solved per block. The likely bottleneck is no longer lack of learning mechanisms. It is temporal credit/memory, training-loop efficiency, and failure-driven iteration.

The next vertical slice is a compact recurrent terminal, preferably a minGRU-style terminal, integrated as a ReCoN ForceProposal specialist rather than as a direct action bypass. ReCoN must still arbitrate and enforce safety.

## Primary Hypothesis

N-link cartpole failures are often temporal, oscillatory, and coupled. Frame stacking is a blunt substitute for memory. A compact recurrent terminal should learn force timing, oscillation damping, and delayed coupling effects better than the current feedforward MLP/policy terminal while preserving the ReCoN graph/arbitration scaffold.

## Do Not Optimize For

- Another open-ended 30-hour training run.
- More unrelated reward knobs.
- A solved claim from a showcase trace or a single friendly seed block.
- A policy that bypasses ReCoN arbitration and safety.

## Implemented Baseline To Preserve

The repo already includes:

- ReCoN graph control.
- Edge plasticity.
- Bandit routing.
- Slow consolidation.
- Learnable node/regime parameters.
- Simple MLP terminal.
- PPO policy terminal.
- Hard-seed training with worker desynchronization.
- Normalized raw observations.
- Frame-stacked observations.
- Training-only success/failure reward wrappers.
- Iterative policy-terminal experiments with multiblock validation.

Treat this as the baseline machinery. The next work should diagnose and improve temporal control.

## Vertical Slice 1: minGRU Terminal

Create:

```text
src/recon_cartpole/recon/mingru_terminal.py
```

Add mode:

```text
recon_mingru_terminal
```

Requirements:

- PyTorch optional dependency only.
- Disabled by default.
- Emits a `ForceProposal`, not a final action.
- ReCoN still arbitrates final force/action.
- `avoid_rail` and existing safety logic must be able to override.
- NaN/invalid outputs are ignored.
- Low-confidence outputs are downweighted.
- Hidden state resets per episode.

Inputs:

- `env` observation mode.
- `normalized_raw` observation mode.
- Configurable sequence length.
- Optional previous action/force.
- Optional selected regime, risk, and urgency.

Outputs/debug fields:

- Force proposal or discrete action logits.
- Confidence.
- Value estimate.
- Optional failure probability estimate.
- `hidden_norm`.
- `sequence_length`.
- Raw force/logits.
- Checkpoint path.

Default architecture:

- Hidden size 32 or 64.
- One minimal recurrent block.
- Small policy head.
- Small value head.
- CPU-friendly.

Candidate config fields:

```text
mingru_enabled
mingru_hidden_size
mingru_sequence_length
mingru_observation_mode
mingru_include_prev_action
mingru_blend
mingru_scope
mingru_checkpoint_path
mingru_confidence_floor
mingru_safety_override
```

## Vertical Slice 2: ReCoN Integration

Integrate minGRU preferably under `stabilize_chain`, or as a sibling regime named `mingru_stabilize_chain`.

The minGRU proposal must flow through the same proposal scoring/arbitration path as other proposals:

- Edge weight.
- Proposal edge weight.
- Bandit multiplier.
- Selected/non-selected multiplier.
- Confidence/urgency.
- Arbitration.

Do not bypass ReCoN.

Trace fields must show:

- minGRU proposal force.
- Final blended force.
- Confidence.
- Hidden norm.
- Value estimate.
- Failure probability.
- Whether safety fallback overrode it.
- Active ReCoN weights used in scoring.

## Vertical Slice 3: Fast Recurrent Training Ladder

Create:

```text
scripts/train_recurrent_terminal_ladder.py
```

The ladder should be fail-fast:

- Train for short blocks.
- Evaluate after each block.
- Kill bad configs early.
- Keep top K checkpoints.
- Retry a small, explicit config sweep.
- Produce one report per candidate.
- Produce a final leaderboard.

Suggested default budget:

- Block size: 100k-250k env steps.
- First gate after 250k steps.
- Kill if mean survival is below a configured floor.
- Promote only if validation improves.
- Default max budget 1M, extended budget 5M.

Training ladder stages:

1. Pure feedforward PPO sanity baseline.
2. minGRU imitation from PPO/ReCoN/best-controller traces.
3. Frozen minGRU inside ReCoN.
4. minGRU inside ReCoN with optional policy-gradient fine-tune.
5. minGRU plus ReCoN learning around it, reported separately.

## Vertical Slice 4: Dataset And Supervised Training

Create:

```text
scripts/build_policy_dataset.py
scripts/train_mingru_supervised.py
```

Dataset examples should include:

- Observation sequence.
- Previous action/force.
- Teacher action.
- Teacher force.
- Reward-to-go.
- Stability delta over k steps.
- Failure-within-k label.
- Seed.
- Controller source.

Teacher sources:

- Pure PPO checkpoint.
- `recon_policy_terminal` checkpoint.
- Heuristic/gain-search controller.
- Mixed failure traces.

Training losses:

- Action imitation loss.
- Value regression loss.
- Failure-prediction BCE.
- Optional confidence calibration loss.

## Vertical Slice 5: Environment Learnability Diagnosis

Before blaming ReCoN, establish whether the exact N=4 env is learnable. For each target env config, compare:

- `baseline_heuristic`.
- Pure feedforward PPO.
- Pure minGRU policy.
- `recon_policy_terminal`.
- `recon_mingru_terminal`.
- `recon_mingru_terminal` plus ReCoN learning.

Report:

- Mean survival.
- p10 survival.
- success@500.
- Max survival.
- Train env steps.
- Wall-clock.
- Eval episodes.
- Seed split.
- Config hash.

Decision rules:

- If pure PPO/minGRU cannot learn N=4, flag environment/training setup as likely bottleneck.
- If pure PPO/minGRU can learn but ReCoN terminal cannot, inspect ReCoN integration/arbitration.
- If ReCoN+minGRU beats pure minGRU on held-out seeds, that is the strongest result.

## Failure-Driven Debugging

For every failed validation block, classify failures:

- `rail_left`.
- `rail_right`.
- `pole_i_angle`.
- `pole_i_velocity`.
- `bad_force_sign`.
- `force_oscillation`.
- `overcorrection`.
- `undercorrection`.
- `minGRU_low_confidence`.
- `minGRU_high_confidence_wrong`.
- `ReCoN_arbitration_overrode_good_policy`.
- `ReCoN_arbitration_followed_bad_policy`.

For minGRU-specific failures, log:

- Hidden norm over time.
- Confidence over time.
- Value estimate over time.
- Predicted failure probability over time.
- Policy force vs final ReCoN force.
- Selected regime.
- Proposal scores.

## Ablations And Final Report

For N=3 and N=4, run same-seed ablations:

- `baseline_heuristic`.
- `static_recon`.
- `recon_fast_bandit`.
- `recon_slow_no_gain_search`.
- `recon_mlp_terminal`.
- `pure_ppo`.
- `pure_mingru_policy`.
- `recon_policy_terminal`.
- `recon_mingru_terminal_frozen`.
- `recon_mingru_terminal_finetuned`.
- `recon_mingru_terminal_plus_recon_learning`.

Each row must include active mechanisms:

- Edge plasticity.
- Bandit.
- Slow consolidation.
- Node-param learning.
- Gain mutation.
- Feedforward policy terminal.
- minGRU terminal.
- Pure PPO.

And metrics:

- Train steps.
- Wall-clock.
- Eval episodes.
- Mean survival.
- p10 survival.
- success@500.
- Max survival.
- Final checkpoint.

## Stop Conditions

Solved N=3:

- held-out test episodes >= 300
- mean survival >= 475 / 500
- p10 survival >= 400 / 500
- success@500 >= 0.80

Solved N=4:

- held-out test episodes >= 300
- mean survival >= 475 / 500
- p10 survival >= 350 / 500
- success@500 >= 0.70

Do not claim solved unless test thresholds pass.

If not solved, report:

- Best checkpoint.
- Best/median/worst replay.
- Failure taxonomy.
- Config leaderboard.
- Next bottleneck hypothesis.

## Acceptance Criteria

- `recon_mingru_terminal` exists and runs inside ReCoN as a `ForceProposal` terminal.
- A supervised minGRU training path exists.
- A fail-fast recurrent training ladder exists.
- The exact N=4 environment has a pure PPO/minGRU learnability report.
- ReCoN+minGRU is compared against pure PPO/minGRU and existing policy terminal.
- Reports clearly say whether minGRU helped, whether ReCoN arbitration helped, and whether the environment itself appears learnable.

## First Implementation Order

1. Add `mingru_terminal.py` with a minimal optional-torch recurrent terminal and unit tests.
2. Integrate it into `ReConCartPoleController` as a ForceProposal under ReCoN arbitration.
3. Add trace/debug fields.
4. Add dataset builder for sequence imitation.
5. Add supervised minGRU trainer.
6. Add fail-fast ladder runner and leaderboard output.
7. Run small N=3 smoke tests before spending time on N=4.
