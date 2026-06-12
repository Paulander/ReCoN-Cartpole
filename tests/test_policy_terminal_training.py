from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_trainer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_policy_terminal.py"
    spec = spec_from_file_location("train_policy_terminal", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_iterative_trainer():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    path = scripts_dir / "train_policy_terminal_iterative.py"
    sys.path.insert(0, str(scripts_dir))
    spec = spec_from_file_location("train_policy_terminal_iterative", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = _load_trainer()
make_env = trainer.make_env
iterative_trainer = _load_iterative_trainer()


def _args(success_bonus: float):
    return SimpleNamespace(
        n_poles=1,
        horizon=1,
        dt=0.02,
        dynamics_mode="parallel",
        action_mode="discrete",
        discrete_action_bins=2,
        force_mag=10.0,
        initial_angle_range=0.0,
        force_noise=0.0,
        link_coupling=0.35,
        hard_train_seeds="",
        hard_train_seed_probability=1.0,
        frame_stack=1,
        policy_observation_mode="env",
        success_bonus=success_bonus,
        failure_penalty=0.0,
    )


def test_success_bonus_is_training_only():
    train_env = make_env(_args(7.0), reward_mode="survival", use_success_bonus=True)
    train_env.reset(seed=1)
    _obs, reward, terminated, truncated, info = train_env.step(1)
    assert not terminated
    assert truncated
    assert reward == 8.0
    assert info["success_bonus"] == 7.0

    eval_env = make_env(_args(7.0), reward_mode="survival", use_success_bonus=False)
    eval_env.reset(seed=1)
    _obs, reward, terminated, truncated, info = eval_env.step(1)
    assert not terminated
    assert truncated
    assert reward == 1.0
    assert "success_bonus" not in info


def test_teacher_action_anchor_penalizes_early_mismatch():
    import gymnasium as gym
    import numpy as np

    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(-10.0, 10.0, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Discrete(5)

        def __init__(self):
            self.config = SimpleNamespace(
                n_poles=1,
                horizon=10,
                x_threshold=2.4,
                theta_threshold_radians=0.2,
                action_mode="discrete",
                discrete_action_bins=5,
                force_mag=10.0,
            )
            self.steps = 0

        def reset(self, *, seed=None, options=None):
            self.steps = 0
            return np.zeros(4, dtype=np.float32), {"raw_state": [0.0, 0.0, 0.0, 0.0]}

        def step(self, action):
            self.steps += 1
            return np.zeros(4, dtype=np.float32), 1.0, False, self.steps >= 10, {"raw_state": [0.0, 0.0, 0.0, 0.0]}

    class Teacher:
        def predict(self, obs, deterministic=True):
            return 2, None

    env = trainer.TeacherActionAnchorWrapper(
        TinyEnv(),
        model_path="unused.zip",
        penalty=0.25,
        observation_mode="env",
        until_fraction=0.5,
        risk_threshold=1.0,
        teacher_model=Teacher(),
    )
    env.reset(seed=1)
    _obs, reward, _terminated, _truncated, info = env.step(4)
    assert reward == 0.75
    assert info["teacher_action_penalty"] == 0.25
    _obs, reward, _terminated, _truncated, info = env.step(2)
    assert reward == 1.0
    assert "teacher_action_penalty" not in info


def test_hard_seed_wrapper_offsets_sampling_by_worker_seed():
    import gymnasium as gym
    import numpy as np

    class SeedRecordingEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(1)

        def __init__(self):
            self.reset_seeds = []

        def reset(self, *, seed=None, options=None):
            self.reset_seeds.append(seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 0.0, False, True, {}

    env_a = SeedRecordingEnv()
    env_b = SeedRecordingEnv()
    wrapper_a = trainer.HardSeedResetWrapper(env_a, [100, 200, 300], probability=1.0)
    wrapper_b = trainer.HardSeedResetWrapper(env_b, [100, 200, 300], probability=1.0)

    wrapper_a.reset(seed=1)
    wrapper_b.reset(seed=2)

    assert env_a.reset_seeds == [200]
    assert env_b.reset_seeds == [300]


def test_failure_penalty_subtracts_only_on_termination():
    import gymnasium as gym
    import numpy as np

    class TerminatingEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 1.0, True, False, {}

    env = trainer.FailurePenaltyWrapper(TerminatingEnv(), 3.5)
    _obs, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert reward == -2.5
    assert info["failure_penalty"] == 3.5


def test_validation_seed_starts_expand_to_disjoint_blocks():
    args = SimpleNamespace(
        validation_seed_start=10,
        validation_seed_starts=[100, 200],
        validation_episodes=3,
    )

    assert iterative_trainer.validation_seeds(args) == [100, 101, 102, 200, 201, 202]


def test_validation_seed_start_falls_back_to_single_block():
    args = SimpleNamespace(
        validation_seed_start=10,
        validation_seed_starts=None,
        validation_episodes=3,
    )

    assert iterative_trainer.validation_seeds(args) == [10, 11, 12]


def test_score_weights_are_configurable():
    summary = {"mean_survival": 100.0, "p10_survival": 40.0, "success_rate": 0.5}

    assert iterative_trainer.score(summary) == 135.0
    assert iterative_trainer.score(
        summary, mean_weight=0.0, p10_weight=0.0, success_weight=200.0
    ) == 100.0


def test_tail_curriculum_scores_lower_tail_and_success():
    tail = _load_script("train_policy_terminal_tail_curriculum")
    summary = tail.tail_metrics([500, 500, 400, 300, 250], horizon=500, cvar_fraction=0.4)

    assert summary["success_rate"] == 0.4
    assert summary["cvar_survival"] == 275.0
    assert tail.tail_score(
        summary,
        mean_weight=0.0,
        p10_weight=0.0,
        cvar_weight=1.0,
        success_weight=100.0,
    ) == 315.0


def test_tail_curriculum_selects_near_miss_tail_seeds():
    tail = _load_script("train_policy_terminal_tail_curriculum")
    summary = {
        "per_seed": [
            {"seed": 1, "steps": 499, "success": False},
            {"seed": 2, "steps": 200, "success": False},
            {"seed": 3, "steps": 500, "success": True},
            {"seed": 4, "steps": 450, "success": False},
        ]
    }

    assert tail.tail_seed_pool(summary, limit=2, min_steps=300) == [1, 4]


def test_tail_curriculum_promotion_rejects_tail_regression():
    tail = _load_script("train_policy_terminal_tail_curriculum")
    args = SimpleNamespace(max_success_regression=0.01, max_p10_regression=5.0, max_cvar_regression=8.0, promotion_mode="score")
    best = {
        "score": 1000.0,
        "validation": {"success_rate": 0.72, "p10_survival": 440.0},
    }
    row = {
        "score": 1200.0,
        "validation": {"success_rate": 0.70, "p10_survival": 450.0},
    }

    assert tail.should_promote(row, best, args) is False


def test_iterative_training_env_enables_hard_seed_wrapper(monkeypatch):
    calls = {}

    def fake_make_env(args, *, reward_mode, use_hard_seeds):
        calls["reward_mode"] = reward_mode
        calls["use_hard_seeds"] = use_hard_seeds
        return "env"

    monkeypatch.setattr(iterative_trainer, "make_env", fake_make_env)
    args = SimpleNamespace(reward_mode="upright_shaping")

    assert iterative_trainer.make_training_env(args) == "env"
    assert calls == {"reward_mode": "upright_shaping", "use_hard_seeds": True}



def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recurrent_terminal_scripts_import_and_hash_configs():
    dataset_builder = _load_script("build_policy_dataset")
    supervised = _load_script("train_mingru_supervised")
    ladder = _load_script("train_recurrent_terminal_ladder")
    autonomous = _load_script("run_n4_autonomous_recurrent")
    pole1 = _load_script("run_n4_pole1_robustness")
    final_gap = _load_script("run_n4_final_success_gap")
    pole1_finetune = _load_script("run_n4_pole1_policy_finetune")
    tail_curriculum = _load_script("train_policy_terminal_tail_curriculum")
    recurrent_tail = _load_script("train_recurrent_policy_terminal_tail_curriculum")
    ppo_sweep = _load_script("run_ppo_sweep")
    residual = _load_script("train_residual_policy_terminal")
    recurrent_curriculum = _load_script("train_recurrent_policy_terminal_curriculum")
    action_compare = _load_script("compare_policy_actions")

    assert callable(dataset_builder.collect)
    assert callable(supervised.train)
    assert ladder.config_hash({"a": 1}) == ladder.config_hash({"a": 1})
    assert ladder.config_hash({"a": 1}) != ladder.config_hash({"a": 2})
    assert autonomous.mechanism_flags("recon_mingru_terminal_plus_recon_learning")["edge_plasticity"] is True
    assert callable(pole1.compare_outcomes)
    assert callable(final_gap.candidate_configs)
    assert callable(pole1_finetune.collect_failure_dataset)
    assert callable(tail_curriculum.run_tail_curriculum)
    assert callable(recurrent_tail.run_recurrent_tail_curriculum)
    assert callable(ppo_sweep.run_sweep)
    assert callable(residual.train_residual)
    assert callable(recurrent_curriculum.run_curriculum)
    assert callable(action_compare.run_comparison)


def test_action_comparison_summarizes_seed_deltas():
    action_compare = _load_script("compare_policy_actions")
    rows = [
        {
            "seed": 1,
            "a_steps": 499,
            "b_steps": 500,
            "delta_steps": 1,
            "a_success": False,
            "b_success": True,
            "a_failure": "pole_1_angle",
            "b_failure": "success",
            "same_outcome": False,
            "action_diff_count": 1,
            "first_action_diff": {"step": 498},
        },
        {
            "seed": 2,
            "a_steps": 500,
            "b_steps": 498,
            "delta_steps": -2,
            "a_success": True,
            "b_success": False,
            "a_failure": "success",
            "b_failure": "rail_right",
            "same_outcome": False,
            "action_diff_count": 2,
            "first_action_diff": {"step": 100},
        },
    ]

    summary = action_compare.summarize_comparison(rows, horizon=500)

    assert summary["success_gain_count"] == 1
    assert summary["success_loss_count"] == 1
    assert summary["changed_seed_count"] == 2
    assert summary["first_diff_step_median"] == 299.0


def test_late_survival_bonus_starts_at_threshold():
    import gymnasium as gym
    import numpy as np
    from types import SimpleNamespace

    class CountingEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(1)

        def __init__(self):
            self.config = SimpleNamespace(horizon=4)

        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 1.0, False, False, {}

    env = trainer.LateSurvivalBonusWrapper(CountingEnv(), bonus=0.5, start_fraction=0.75)
    env.reset(seed=1)
    _obs, reward_1, *_rest, info_1 = env.step(0)
    _obs, reward_2, *_rest, info_2 = env.step(0)
    _obs, reward_3, *_rest, info_3 = env.step(0)

    assert reward_1 == 1.0
    assert reward_2 == 1.0
    assert reward_3 == 1.5
    assert "late_survival_bonus" not in info_1
    assert "late_survival_bonus" not in info_2
    assert info_3["late_survival_bonus"] == 0.5


def test_policy_observation_wrapper_pads_n3_to_raw4_prev_force():
    args = _args(0.0)
    args.n_poles = 3
    args.horizon = 2
    args.policy_observation_mode = "normalized_raw4_prev_force"
    env = make_env(args, reward_mode="survival", use_success_bonus=False)
    obs, _info = env.reset(seed=4)

    assert obs.shape == (11,)
    assert obs[5] == 0.0
    assert obs[9] == 0.0
    assert obs[-1] == 0.0


def test_observation_normalizer_wrapper_applies_stats(tmp_path):
    import gymnasium as gym
    import json
    import numpy as np

    class ToyEnv(gym.Env):
        observation_space = gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            return np.asarray([3.0, 6.0], dtype=np.float32), {}

        def step(self, action):
            return np.asarray([3.0, 6.0], dtype=np.float32), 0.0, False, True, {}

    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"mean": [1.0, 0.0], "var": [4.0, 9.0], "epsilon": 0.0, "clip_obs": 10.0}), encoding="utf-8")
    env = trainer.ObservationNormalizerWrapper(ToyEnv(), str(path))
    obs, _info = env.reset(seed=0)

    assert np.allclose(obs, [1.0, 2.0])


def test_tail_curriculum_lexicographic_promotion_prefers_success_over_score():
    tail = _load_script("train_policy_terminal_tail_curriculum")
    args = SimpleNamespace(
        max_success_regression=0.0,
        max_p10_regression=2.0,
        max_cvar_regression=2.0,
        promotion_mode="lexicographic_success",
    )
    best = {
        "score": 900.0,
        "validation": {
            "success_rate": 0.696,
            "p10_survival": 434.9,
            "cvar_survival": 414.8,
            "mean_survival": 484.9,
        },
    }
    row = {
        "score": 897.0,
        "validation": {
            "success_rate": 0.700,
            "p10_survival": 434.0,
            "cvar_survival": 414.8,
            "mean_survival": 484.9,
        },
    }

    assert tail.should_promote(row, best, args) is True


def test_tail_curriculum_final_seed_starts_expand_blocks():
    tail = _load_script("train_policy_terminal_tail_curriculum")
    args = SimpleNamespace(final_seed_start=10, final_seed_starts=[100, 200], final_eval_episodes=2)

    assert tail.final_eval_seeds(args) == [100, 101, 200, 201]
