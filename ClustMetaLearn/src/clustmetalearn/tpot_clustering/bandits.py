"""Multi-armed bandits for adaptive algorithm selection (UCB1, Softmax)."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def normalize_reward(fitness: float, bad_threshold: float = -1e8) -> float:
    """Map raw fitness to [0, 1] for bandit updates."""
    if not math.isfinite(fitness) or fitness <= bad_threshold + 1.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-fitness)))


@dataclass
class ArmStats:
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.reward_sum / self.pulls


class MultiArmedBandit(ABC):
    """Base bandit over arms 0 .. n_arms-1."""

    def __init__(self, n_arms: int, *, rng: random.Random | None = None):
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self.n_arms = n_arms
        self.rng = rng or random.Random()
        self.arms: list[ArmStats] = [ArmStats() for _ in range(n_arms)]

    @property
    def total_pulls(self) -> int:
        return sum(a.pulls for a in self.arms)

    def update(self, arm: int, reward: float) -> None:
        arm = int(arm)
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"arm {arm} out of range [0, {self.n_arms})")
        r = float(np.clip(reward, 0.0, 1.0))
        self.arms[arm].pulls += 1
        self.arms[arm].reward_sum += r

    def update_from_fitness(self, arm: int, fitness: float, *, bad_threshold: float = -1e8) -> None:
        self.update(arm, normalize_reward(fitness, bad_threshold=bad_threshold))

    @abstractmethod
    def select_arm(self) -> int:
        """Choose an arm to play next."""

    def mean_rewards(self) -> list[float]:
        return [a.mean_reward for a in self.arms]


class UCB1Bandit(MultiArmedBandit):
    """UCB1 arm selection."""

    def __init__(self, n_arms: int, *, exploration_c: float = math.sqrt(2.0), rng=None):
        super().__init__(n_arms, rng=rng)
        self.exploration_c = exploration_c

    def select_arm(self) -> int:
        for arm in range(self.n_arms):
            if self.arms[arm].pulls == 0:
                return arm
        t = max(self.total_pulls, 1)
        log_t = math.log(t)
        best_arm = 0
        best_score = -math.inf
        for arm in range(self.n_arms):
            n_i = self.arms[arm].pulls
            mean = self.arms[arm].mean_reward
            bonus = self.exploration_c * math.sqrt(log_t / n_i)
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm


class SoftmaxBandit(MultiArmedBandit):
    """Softmax (Boltzmann) arm selection."""

    def __init__(self, n_arms: int, *, temperature: float = 1.0, rng=None):
        super().__init__(n_arms, rng=rng)
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def select_arm(self) -> int:
        unplayed = [a for a in range(self.n_arms) if self.arms[a].pulls == 0]
        if unplayed:
            return self.rng.choice(unplayed)

        means = np.array([a.mean_reward for a in self.arms], dtype=float)
        scaled = means / self.temperature
        scaled -= np.max(scaled)
        exp_v = np.exp(scaled)
        probs = exp_v / exp_v.sum()
        idx = self.rng.choices(range(self.n_arms), weights=probs.tolist(), k=1)[0]
        return int(idx)


def make_bandit(
    policy: str,
    n_arms: int,
    *,
    rng: random.Random | None = None,
    exploration_c: float = math.sqrt(2.0),
    temperature: float = 1.0,
) -> MultiArmedBandit | None:
    if policy == "none":
        return None
    if policy == "ucb1":
        return UCB1Bandit(n_arms, exploration_c=exploration_c, rng=rng)
    if policy == "softmax":
        return SoftmaxBandit(n_arms, temperature=temperature, rng=rng)
    raise ValueError(f"Unknown bandit policy: {policy!r}")
