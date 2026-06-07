# Curriculum

Start with sanity before spectacle:

1. `cartpole_v1_static`: compare `baseline_heuristic` and `static_recon`.
2. `nlink_1_static`: verify the custom environment and controller.
3. `nlink_2_fast_bandit`: enable fast plasticity, bandit choice, and modulation.
4. Increase to 3, 4, 5, and 6 poles only after held-out seeds pass the previous stage.

Each ground iteration should freeze config, train, evaluate on fixed seeds, export
best/median/worst traces, render reports, and promote only if validation improves.

