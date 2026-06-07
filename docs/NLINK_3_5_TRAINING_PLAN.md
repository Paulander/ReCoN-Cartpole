# N-link 3/4/5 Training Plan

Goal: produce repeatable first-pass metrics for 3, 4, and 5 linked poles, with artifacts good enough to inspect controller behavior.

Plan:

1. Use `recon_fast_bandit` as the first training mode because it exercises the ReCoN graph, fast plasticity, UCB regime selection, and modulation.
2. Carry bandit priors across training episodes, but freeze all learning during held-out evaluation.
3. Use the same horizon and seed policy for 3, 4, and 5 links so results are comparable.
4. Export `metrics.json`, `summary.md`, and one replay HTML/JSON per link count.
5. Treat results as iteration metrics only. The current N-link dynamics are approximate and need a validated multibody implementation before any high-N claim is meaningful.

Default command:

```bash
uv run python scripts/train_nlink_sweep.py --links 3 4 5 --train-episodes 80 --eval-episodes 25 --horizon 500
```

Promotion criteria for future runs:

- N=3: held-out mean survival >= 300 before tuning N=4 seriously.
- N=4: held-out mean survival >= 250 before tuning N=5 seriously.
- N=5: report progress only; do not claim solved until mean survival >= 475 over 100+ held-out seeds.
