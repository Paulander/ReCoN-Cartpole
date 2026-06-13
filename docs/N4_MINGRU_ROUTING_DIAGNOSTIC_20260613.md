# N=4 minGRU Routing Diagnostic - 2026-06-13

## Why

The latest mixed minGRU PPO run showed a small gap between pure minGRU evaluation and ReCoN-routed evaluation. That suggested the bottleneck might be ReCoN arbitration/routing rather than recurrent policy capacity.

A new diagnostic script, `scripts/compare_mingru_routing.py`, compares the same minGRU checkpoint in two closed-loop modes on identical seeds:

- `pure_minGRU_policy`: terminal force is converted directly to the 5-bin action;
- `ReCoN_routed_minGRU`: terminal is inserted into the ReCoN proposal/arbitration path.

The report records first action differences with raw state, pure terminal confidence/logit margin, selected ReCoN regime, winning proposal, suppressed proposals, and minGRU diagnostic fields.

## Incumbent Checkpoint

`reports/n4_mingru_dagger9_fresh_option_aux_20260613_seed9131k/supervised_mingru/mingru_terminal.pt`

Environment: N=4, serial Lagrange dynamics, `dt=0.0005`, 5 action bins, force magnitude `10`, force noise `0.02`, link coupling `12`, held-out mixed validation seeds.

## Finding: Hard-Select Suppresses the Learned Terminal

On the first held-out block without passthrough:

`reports/n4_mingru_routing_compare_heldout1900k_20260613`

| mode | mean | p10 | success |
| --- | ---: | ---: | ---: |
| pure minGRU | 482.45 | 427.7 | 0.55 |
| ReCoN-routed minGRU | 481.10 | 424.9 | 0.50 |

The diagnostic found one success loss: seed `1900015` succeeds under pure minGRU but fails under ReCoN routing. At the first action difference:

- pure minGRU chose action `4` / force `10.0`;
- ReCoN chose action `0` / force `-10.0`;
- minGRU was available and predicted action `4` with confidence `0.9822`;
- `hard_select` selected `recover_worst_pole`, so the `stabilize_chain` proposal containing the minGRU force was suppressed;
- the winning proposal was `recover_worst_pole`, reason `worst pole recovery`.

This pattern appears repeatedly: the terminal prediction is often present and confident, but the selected heuristic regime can win before the learned action reaches final control.

## Passthrough Check

With existing minGRU passthrough enabled:

`reports/n4_mingru_routing_compare_mixed80_passthrough_20260613`

| mode | mean | p10 | success |
| --- | ---: | ---: | ---: |
| pure minGRU | 486.725 | 451.9 | 0.6875 |
| ReCoN-routed minGRU + passthrough | 486.725 | 451.9 | 0.6875 |

Across all 80 held-out mixed seeds:

- changed seeds: `0`;
- action-changed seeds: `0`;
- success gains: `0`;
- success losses: `0`.

So passthrough exactly restores the pure terminal behavior for this checkpoint/configuration.

## Interpretation

The current recurrent plateau is not explained by a pure policy vs ReCoN metric mismatch anymore. With passthrough enabled, ReCoN can faithfully execute the learned terminal. Without passthrough, `hard_select` can suppress learned behavior whenever another regime wins selection.

This means future minGRU PPO/curriculum reports should either:

- evaluate with passthrough enabled when the claim is about the learned terminal policy; or
- explicitly state that the metric includes hard-select ReCoN arbitration losses.

## Next Step

Use passthrough-enabled guarded evaluation for the next minGRU PPO/curriculum iteration, then focus on improving the actual pure/recurrent terminal above the current `0.6875` held-out success plateau. A more ReCoN-faithful alternative is to learn or gate regime selection so `stabilize_chain` is not suppressed when the minGRU has high-confidence corrective action, but passthrough is the immediate controlled baseline.
