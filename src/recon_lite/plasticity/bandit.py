from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BanditArmState:
    child_id: str
    pulls: int = 0
    sum_reward: float = 0.0
    sum_sq_reward: float = 0.0
    last_reward: float = 0.0

    def mean_reward(self) -> float:
        return self.sum_reward / self.pulls if self.pulls else 0.0

    def variance(self) -> float:
        if self.pulls < 2:
            return 0.0
        mean = self.mean_reward()
        return self.sum_sq_reward / self.pulls - mean * mean


@dataclass
class BanditConfig:
    enabled: bool = True
    c_explore: float = 1.0
    min_pulls_before_ucb: int = 1


BanditStateDict = Dict[str, Dict[str, BanditArmState]]


def init_bandit_state(parent_children: Dict[str, List[str]]) -> BanditStateDict:
    return {
        parent_id: {child_id: BanditArmState(child_id) for child_id in children}
        for parent_id, children in parent_children.items()
    }


def ucb_score(
    arm: BanditArmState,
    total_pulls: int,
    c_explore: float,
    epsilon: float = 1e-6,
) -> float:
    if arm.pulls == 0:
        return float("inf")
    bonus = c_explore * math.sqrt(2.0 * math.log(total_pulls + 1) / (arm.pulls + epsilon))
    return arm.mean_reward() + bonus


def choose_child(
    parent_id: str,
    state: BanditStateDict,
    c_explore_eff: float,
    config: BanditConfig,
) -> Optional[str]:
    if not config.enabled or parent_id not in state:
        return None
    arms = state[parent_id]
    for child_id, arm in arms.items():
        if arm.pulls < config.min_pulls_before_ucb:
            return child_id
    total_pulls = sum(arm.pulls for arm in arms.values())
    scores: List[Tuple[str, float]] = [
        (child_id, ucb_score(arm, total_pulls, c_explore_eff))
        for child_id, arm in arms.items()
    ]
    return max(scores, key=lambda item: item[1])[0] if scores else None


def assign_reward(parent_id: str, child_id: str, reward: float, state: BanditStateDict) -> bool:
    arm = state.get(parent_id, {}).get(child_id)
    if arm is None:
        return False
    arm.pulls += 1
    arm.sum_reward += reward
    arm.sum_sq_reward += reward * reward
    arm.last_reward = reward
    return True


def reset_bandit_episode(state: BanditStateDict) -> None:
    for arms in state.values():
        for arm in arms.values():
            arm.pulls = 0
            arm.sum_reward = 0.0
            arm.sum_sq_reward = 0.0
            arm.last_reward = 0.0


def snapshot_bandit(state: BanditStateDict) -> Dict[str, Any]:
    return {
        parent_id: {
            child_id: {
                "pulls": arm.pulls,
                "mean_reward": round(arm.mean_reward(), 4),
                "last_reward": round(arm.last_reward, 4),
            }
            for child_id, arm in arms.items()
            if arm.pulls > 0
        }
        for parent_id, arms in state.items()
        if any(arm.pulls > 0 for arm in arms.values())
    }


@dataclass
class BanditPriors:
    arm_stats: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    total_episodes: int = 0
    decay_factor: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_stats": self.arm_stats,
            "total_episodes": self.total_episodes,
            "decay_factor": self.decay_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BanditPriors":
        return cls(
            arm_stats=data.get("arm_stats", {}),
            total_episodes=int(data.get("total_episodes", 0)),
            decay_factor=float(data.get("decay_factor", 0.9)),
        )


def init_bandit_state_with_priors(
    parent_children: Dict[str, List[str]],
    priors: Optional[BanditPriors] = None,
    prior_weight: float = 0.5,
) -> BanditStateDict:
    state = init_bandit_state(parent_children)
    if not priors or prior_weight <= 0:
        return state
    for parent_id, arms in state.items():
        for child_id, arm in arms.items():
            prior = priors.arm_stats.get(parent_id, {}).get(child_id)
            if prior:
                pulls = int(prior.get("pulls", 0) * prior_weight)
                mean = float(prior.get("mean_reward", 0.0))
                arm.pulls = pulls
                arm.sum_reward = mean * pulls
                arm.sum_sq_reward = mean * mean * pulls
    return state


def export_priors(state: BanditStateDict) -> BanditPriors:
    priors = BanditPriors(total_episodes=1)
    for parent_id, arms in state.items():
        priors.arm_stats[parent_id] = {}
        for child_id, arm in arms.items():
            if arm.pulls:
                priors.arm_stats[parent_id][child_id] = {
                    "pulls": float(arm.pulls),
                    "sum_reward": round(arm.sum_reward, 4),
                    "mean_reward": round(arm.mean_reward(), 4),
                }
    return priors


def merge_priors(old_priors: BanditPriors, new_priors: BanditPriors, decay: float = 0.9) -> BanditPriors:
    merged = BanditPriors(decay_factor=decay)
    for parent_id in set(old_priors.arm_stats) | set(new_priors.arm_stats):
        merged.arm_stats[parent_id] = {}
        old_arms = old_priors.arm_stats.get(parent_id, {})
        new_arms = new_priors.arm_stats.get(parent_id, {})
        for child_id in set(old_arms) | set(new_arms):
            old = old_arms.get(child_id, {})
            new = new_arms.get(child_id, {})
            pulls = old.get("pulls", 0.0) * decay + new.get("pulls", 0.0)
            reward_sum = old.get("sum_reward", 0.0) * decay + new.get("sum_reward", 0.0)
            if pulls > 0:
                merged.arm_stats[parent_id][child_id] = {
                    "pulls": round(pulls, 2),
                    "sum_reward": round(reward_sum, 4),
                    "mean_reward": round(reward_sum / pulls, 4),
                }
    merged.total_episodes = old_priors.total_episodes + new_priors.total_episodes
    return merged


def save_priors(priors: BanditPriors, path: str) -> None:
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(priors.to_dict(), indent=2), encoding="utf-8")


def load_priors(path: str) -> BanditPriors:
    import json
    from pathlib import Path

    return BanditPriors.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

