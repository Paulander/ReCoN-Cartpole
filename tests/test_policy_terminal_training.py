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

    assert callable(dataset_builder.collect)
    assert callable(supervised.train)
    assert ladder.config_hash({"a": 1}) == ladder.config_hash({"a": 1})
    assert ladder.config_hash({"a": 1}) != ladder.config_hash({"a": 2})
    assert autonomous.mechanism_flags("recon_mingru_terminal_plus_recon_learning")["edge_plasticity"] is True
    assert callable(pole1.compare_outcomes)
