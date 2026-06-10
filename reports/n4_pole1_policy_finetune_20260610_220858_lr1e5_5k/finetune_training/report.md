# Policy Terminal Training Report

Status: `resumed`
Reward mode: `upright_shaping`
Model: `reports/n4_pole1_policy_finetune_20260610_220858_lr1e5_5k/finetune_training/ppo_policy_terminal.zip`
Train timesteps: `5000`
PPO config: `{'policy': 'MlpPolicy', 'net_arch': '64,64', 'activation': 'tanh', 'learning_rate': 1e-05, 'n_steps': 1024, 'batch_size': 64, 'n_epochs': 2, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.04, 'ent_coef': 0.0, 'vf_coef': 0.5, 'max_grad_norm': 0.5, 'frame_stack': 1, 'policy_observation_mode': 'normalized_raw', 'success_bonus': 25.0, 'failure_penalty': 2.0, 'vec_env': 'dummy'}`
Policy terminal scope: `stabilize_chain`
Frame stack: `1`
Policy observation mode: `normalized_raw`
Success bonus: `25.0`
Failure penalty: `2.0`
Vec env: `dummy`
Wall-clock seconds: `95.69`

| evaluator | mean | p10 | success | max | episodes |
|---|---:|---:|---:|---:|---:|
| pure_ppo | 460.0 | 344.9 | 0.61 | 500.0 | 80 |
| recon_policy_terminal | 488.1 | 450.0 | 0.72 | 500.0 | 80 |

## Claim Discipline

This report separates a learned PPO policy from ReCoN's graph scaffold. It is not pure symbolic ReCoN, and it is not a solved claim unless the held-out solve thresholds are met.
