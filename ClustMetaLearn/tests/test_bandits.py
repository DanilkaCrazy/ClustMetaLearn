import random

from clustmetalearn.tpot_clustering.bandits import SoftmaxBandit, UCB1Bandit, make_bandit
from clustmetalearn.tpot_clustering.encoding import N_ALGO_CHOICES, SearchSpace
from clustmetalearn.tpot_clustering.evolve import EvolutionConfig, run_evolution
from sklearn.datasets import make_blobs


def test_ucb1_prefers_unplayed_arm():
    b = UCB1Bandit(3, rng=random.Random(0))
    b.update(0, 0.9)
    b.update(0, 0.9)
    assert b.select_arm() in (1, 2)


def test_softmax_selects_valid_arm():
    b = SoftmaxBandit(4, temperature=0.5, rng=random.Random(1))
    for arm in range(4):
        b.update(arm, 0.1 * arm)
    arm = b.select_arm()
    assert 0 <= arm < 4


def test_make_bandit_none():
    assert make_bandit("none", 4) is None


def test_evolution_with_ucb1_bandit():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=4)
    space = SearchSpace.from_shape(X.shape[0], X.shape[1])
    cfg = EvolutionConfig(
        generations=1,
        population_size=10,
        cv_splits=3,
        random_state=2,
        metric="calinski_harabasz",
        bandit="ucb1",
        bandit_bias=0.7,
        ranking_trick=True,
        ranking_warmup=5,
        ranking_oversample=2,
    )
    result = run_evolution(X, None, space, cfg)
    assert result.best_fitness > -1e8
    assert result.bandit_arm_pulls is not None
    assert len(result.bandit_arm_pulls) == N_ALGO_CHOICES
    assert sum(result.bandit_arm_pulls) > 0
