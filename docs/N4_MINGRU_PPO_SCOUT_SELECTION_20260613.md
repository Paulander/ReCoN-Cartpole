# N=4 minGRU PPO Scout Selection - 2026-06-13

## Change

`train_mingru_ppo.py` now supports optional per-iteration held-out scout validation:

- `--scout-eval-episodes N` enables scout evaluation;
- `--scout-seed-starts ...` selects the held-out scout blocks, defaulting to `--final-seed-starts` when omitted;
- `--scout-every-iterations K` controls scout frequency;
- `--select-best-scout-checkpoint` final-validates the best scout checkpoint instead of blindly using the last PPO update.

The default behavior is unchanged when scout evaluation is disabled.

The report now distinguishes:

- `checkpoint_path`: the checkpoint actually final-evaluated/promoted;
- `final_checkpoint_path`: the last PPO update;
- `selected_scout_checkpoint_path`: the chosen scout checkpoint when selection is enabled;
- `scout_history`: per-scout score, metrics, and checkpoint path.

## Why

The previous passthrough PPO run showed that final-update PPO can regress even when KL remains small. Around the current plateau, train batch success is noisy and the best policy may appear transiently before later iterations drift. Scout selection is a guard against losing that transient improvement.

## Smoke Verification

A tiny end-to-end smoke run exercised checkpoint saving, scout validation, scout selection, and final reporting:

`reports/smoke_mingru_ppo_scout_20260613`

The smoke selected `checkpoint_iter_001.pt`, proving the new selected-scout path executes.

## Bounded N=4 Run

Run directory:

`reports/n4_mingru_ppo_scout_select_20260613_seed9230k`

Starting checkpoint:

`reports/n4_mingru_dagger9_fresh_option_aux_20260613_seed9131k/supervised_mingru/mingru_terminal.pt`

Important settings:

- N=4, serial Lagrange dynamics, `dt=0.0005`;
- 5 discrete force bins, force magnitude `10`;
- passthrough-enabled ReCoN evaluation;
- 96 training episodes: 34 hard-tail seeds and 62 fresh seeds;
- 4 PPO iterations, 24 rollout episodes per iteration;
- scout validation after every iteration on 20 held-out mixed seeds;
- final validation on the usual 80 held-out mixed seeds.

Scout scores:

| Iteration | Train success | Scout success | Scout p10 | Scout score |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4167 | 0.65 | 450.1 | 1148.540 |
| 2 | 0.2917 | 0.65 | 450.8 | 1149.225 |
| 3 | 0.4583 | 0.65 | 449.8 | 1148.210 |
| 4 | 0.6250 | 0.65 | 450.2 | 1148.640 |

The selector chose iteration 2:

`reports/n4_mingru_ppo_scout_select_20260613_seed9230k/checkpoint_iter_002.pt`

## Final Held-Out Result

| Model | Mean | P10 | Success |
| --- | ---: | ---: | ---: |
| incumbent pure/ReCoN passthrough | 486.725 | 451.9 | 0.6875 |
| scout-selected candidate | 486.600 | 452.6 | 0.6875 |

Promotion score:

- incumbent: `1188.0725`;
- candidate: `1188.7600`;
- promoted: `true`.

## Interpretation

This is real but narrow progress. The candidate did not increase success rate, so N=4 is still unsolved. It did preserve the incumbent success rate while improving p10 survival enough to promote under the current guarded score.

The run also shows why scout selection matters: iteration 4 had the highest train success, but iteration 2 had the best held-out scout score and became the promoted checkpoint. Train success alone would have picked the wrong iteration.

## Next Step

Use the scout-selected checkpoint as the new recurrent incumbent for further improvement attempts, but keep the no-solve-claim rule. The next likely useful move is to repeat scout-selected PPO with either:

- a slightly larger scout block to reduce selection noise;
- a more conservative update around the promoted checkpoint;
- a gated residual/update head that freezes most of the incumbent and only learns late-failure corrections.
