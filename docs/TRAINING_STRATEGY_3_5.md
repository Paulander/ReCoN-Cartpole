# Training Strategy for 3, 4, and 5 Links

The current 3/4/5 sweep is a baseline measurement, not a solve. It shows that the first ReCoN controller survives tens of steps on the approximate coupled N-link environment, but it does not yet learn a robust high-link controller.

## Non-cheating Rules

The point is to train a ReCoN executive, not to hide a solved controller inside hardcoded branches.

Allowed:

- Domain-general sensor features: cart position/velocity, per-link sin/cos angle, angular velocity, energy-like terms, risk, urgency, and failure margins.
- A small family of simple force-proposal primitives shared across all N, generated from templates.
- Curriculum over initial state difficulty, link count, force noise, and horizon.
- Learning over edge weights, regime selection, proposal blending, and small numeric controller gains.
- Baseline controllers for comparison and teacher rollouts, as long as evaluation reports them separately.

Not allowed for the main ReCoN result:

- Per-N hand-tuned special-case policies like `if n == 5 then use this force formula`.
- Training/evaluation seed leakage.
- Claiming solved from showcase traces.
- Changing termination thresholds during evaluation unless the report names the altered benchmark.
- Reward functions that directly optimize the reported success metric in a way unavailable to the controller at runtime.

## What Should Actually Learn

The first implementation only lets bandit statistics persist during a sweep and applies fast edge plasticity inside episodes. To solve 3/4/5, we need more trainable but still interpretable parameters:

1. Persistent edge weights: consolidate successful fast-plasticity deltas into `w_base` only after held-out validation improves.
2. Regime priors: persistent UCB priors for `avoid_rail`, `damp_energy`, `recover_worst_pole`, `recover_base_pole`, `stabilize_chain`, and `center_cart`.
3. Proposal gains: train a small vector of shared coefficients for the proposal primitives, for example angle gain, velocity gain, energy gain, cart-centering gain, and outer-link weighting. These should be global or smoothly parameterized by link index, not one-off per N.
4. Arbitration weights: learn how much confidence, urgency, and link index should affect final force blending.
5. Curriculum thresholds: learn through staged difficulty, not by relaxing final evaluation.

This is likely CPU-friendly. A 3090 helps only if we batch many environment rollouts, add a differentiable surrogate, or run optional RL/teacher baselines. Plain ReCoN graph ticking plus Python Gym steps will be CPU-bound unless we vectorize environments.

## Environment Controllability

The N-link environment must use actual linked-chain coupling for multi-link stages. The original default `link_coupling=0.35` behaves more like several identical independent poles driven by one cart, which creates near-uncontrollable relative modes. Curriculum stages for N=3 therefore set `link_coupling=12.0`; this is an explicit benchmark parameter, not a hidden controller shortcut.

## Curriculum

Use phases for each N:

1. **Warm start**: short horizon, narrow initial angle range, no force noise.
2. **Stability**: increase horizon to 500 while keeping narrow starts.
3. **Robust starts**: widen initial angles and angular velocities.
4. **Noise**: add force noise and small mass/length perturbations.
5. **Held-out validation**: freeze learning and evaluate on fixed unseen seeds.

Advance only when the held-out phase passes. Do not advance because the best trace looks good.

Suggested promotion gates:

| Stage | Gate |
|---|---:|
| N=3 warm | mean survival >= 200 over 50 held-out seeds |
| N=3 stable | mean survival >= 400 over 100 held-out seeds |
| N=3 solved | mean survival >= 475 and p10 >= 350 over 100 held-out seeds |
| N=4 warm | mean survival >= 175 over 50 held-out seeds |
| N=4 stable | mean survival >= 350 over 100 held-out seeds |
| N=5 warm | mean survival >= 150 over 50 held-out seeds |
| N=5 stable | mean survival >= 300 over 100 held-out seeds |

## Training Loop

For each ground iteration:

1. Freeze graph, environment config, seed split, and trainable parameter schema.
2. Run a training block with persistent priors/consolidation enabled.
3. Freeze learning.
4. Evaluate on held-out seeds.
5. Export best, median, and worst traces.
6. Produce failure taxonomy: rail exit, base-link angle, outer-link divergence, or velocity blow-up.
7. Promote checkpoint only if held-out metrics improve.

## Next Implementation Steps

1. Add checkpointed slow consolidation and validation-based promotion.
2. Add trainable proposal/arbitration gains with bounded random search or CMA-ES style evolution over CPU-parallel rollouts.
3. Add vectorized rollouts so the 3090 is not required, but many CPU cores can help; only consider GPU if adding a neural teacher/baseline or JAX/Torch batched dynamics.
4. Replace the approximate N-link dynamics with a validated multibody model before making scientific claims.
5. Add ablations for each learning mechanism at every promoted checkpoint.

## GPU Position

A 3090 is useful for optional baselines and batched differentiable experiments, but the current architecture probably benefits more from:

- vectorized CPU environments,
- multiprocessing rollouts,
- better dynamics,
- checkpointed slow consolidation,
- and principled parameter search.

If we add a teacher policy, use the GPU to train PPO/SAC or a small neural controller, then compare ReCoN against it or distill only interpretable proposal gains. Keep that optional and reported separately.
