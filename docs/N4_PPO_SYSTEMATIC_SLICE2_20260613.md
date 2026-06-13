# N=4 PPO Systematic Slice 2 - 2026-06-13

## Goal

Continue the systematic PPO sweep on the current 5-bin N=4 setup without making solve claims from train seeds. This slice was designed to cover under-tested continuation axes while using mixed-grid validation and held-out mixed final evaluation.

Base checkpoint:

`reports/policy_terminal_n4_worker_seeded_combined_p0125_lr25e6_seed1520k/checkpoint_025000.zip`

Report:

`reports/n4_ppo_systematic_slice2_20260613_seed9270k`

## Setup

Fixed environment/control settings:

- `n_poles=4`
- `dynamics_mode=serial_lagrange`
- `dt=0.0005`
- `discrete_action_bins=5`
- `force_mag=10`
- `initial_angle_range=0.05`
- `force_noise=0.02`
- `link_coupling=12`
- `selection_mode=hard_select`
- `policy_terminal_scope=stabilize_chain`
- `policy_observation_mode=normalized_raw`

Training geometry:

- start from the current best PPO terminal;
- 2 chunks of 8000 PPO timesteps per candidate;
- hard-seed probability `0.40` from `reports/hard_seeds_n4_combined_nearmiss_600/hard_seeds.txt`;
- validation starts `900000`, `930000`, `970000`, `1010000`, 20 episodes each;
- final held-out starts `1900000`, `2000000`, `2100000`, `2200000`, 10 episodes each;
- GPU was available, but Stable-Baselines warned MLP PPO is usually CPU-friendlier. The run completed on CUDA, with poor expected GPU utilization.

## Candidate Grid

The sweep runner grid includes all requested axes: learning rate, clip range, n_steps, n_epochs, GAE lambda, entropy coefficient, net architecture, VecNormalize, and late-survival bonus. This bounded slice selected four original grid indices from a 768-point focused grid.

| grid | lr | clip | n_steps | epochs | gae | ent | net | VecNormalize | late bonus | selected checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| 415 | 2.5e-7 | 0.001 | 512 | 1 | 0.98 | 0.0 | 128,128 | true | 0.005 | start |
| 469 | 2.5e-7 | 0.001 | 512 | 2 | 0.98 | 0.001 | 64,64 | false | 0.005 | start |
| 575 | 2.5e-7 | 0.001 | 1024 | 2 | 0.98 | 0.001 | 256,128 | true | 0.005 | start |
| 767 | 2.5e-7 | 0.003 | 1024 | 2 | 0.98 | 0.001 | 256,128 | true | 0.005 | start |

## Held-Out Final Results

All four candidates selected `checkpoint_000000_start.zip` as the best checkpoint after validation. Final held-out metrics were therefore identical across rows:

| grid | mean | p10 | cvar | success | episodes | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 415 | 485.65 | 432.9 | 415.75 | 0.700 | 40 | not solved |
| 469 | 485.65 | 432.9 | 415.75 | 0.700 | 40 | not solved |
| 575 | 485.65 | 432.9 | 415.75 | 0.700 | 40 | not solved |
| 767 | 485.65 | 432.9 | 415.75 | 0.700 | 40 | not solved |

Validation was high for the start checkpoint (`success=0.8125`, `p10=465.8`, `mean=491.66` over 80 episodes), but the held-out final block only reached `success=0.700` over 40 episodes. This is encouraging but too small and too narrow for a solve claim.

## Interpretation

This was a plateau slice. The selected continuation updates did not beat the incumbent on validation, even when varying VecNormalize, network size, n_steps, n_epochs, GAE, entropy, and late-survival bonus.

The result reinforces the recent pattern: tiny PPO continuation from the current best mostly preserves behavior rather than improving the N=4 tail. Future PPO effort should not be more of the same micro-continuation. Better next options are:

- change the training distribution more substantially, for example stronger curriculum stage transitions or hard-tail data collection;
- use a recurrent/minGRU curriculum with previous force/history and scout selection;
- train a true recovery specialist with trajectory-level labels instead of one-step residual bins;
- if PPO is continued, use CPU or benchmark CPU vs CUDA because SB3 warned this MLP policy may be slower on GPU.

Current N=4 status remains unsolved. No higher-N solve claims are supported by this run.
