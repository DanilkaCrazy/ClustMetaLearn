"""DEAP evolution loop for clustering pipelines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
from deap import algorithms, base, creator, tools

from clustmetalearn.tpot_clustering.bandits import MultiArmedBandit, make_bandit
from clustmetalearn.tpot_clustering.ranking import PairwiseLinearRanker, RankingTrickConfig
from clustmetalearn.tpot_clustering.encoding import (
    N_ALGO_CHOICES,
    SearchSpace,
    individual_to_pipeline,
)
from clustmetalearn.tpot_clustering.fitness import MetricName, evaluate_individual

BanditPolicy = Literal["none", "ucb1", "softmax"]
_ALGO_GENE = 3


def _register_deap_types() -> None:
    if not hasattr(creator, "FitnessMaxCluster"):
        creator.create("FitnessMaxCluster", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualCluster"):
        creator.create("IndividualCluster", list, fitness=creator.FitnessMaxCluster)


def _mutate(
    individual: list,
    space: SearchSpace,
    indpb: float,
    *,
    bandit: MultiArmedBandit | None,
    bandit_bias: float,
    rng: random.Random,
) -> tuple:
    for i in range(len(individual)):
        if rng.random() >= indpb:
            continue
        if i in (0, 1):
            individual[i] = rng.randint(0, 1)
        elif i == _ALGO_GENE:
            if bandit is not None and rng.random() < bandit_bias:
                individual[i] = bandit.select_arm()
            else:
                individual[i] = rng.randint(0, N_ALGO_CHOICES - 1)
        elif i == 2:
            individual[i] = rng.randint(2, space.max_pca)
        elif i == 4:
            individual[i] = rng.randint(2, space.max_k)
        elif i == 5:
            individual[i] = rng.randint(0, 2)
    space.clip(individual)
    return (individual,)


def _evaluate_and_update_bandit(
    ind,
    toolbox_eval,
    bandit: MultiArmedBandit | None,
    bad_threshold: float,
):
    fit = toolbox_eval(ind)
    ind.fitness.values = fit
    if bandit is not None and fit[0] > bad_threshold + 1.0:
        bandit.update_from_fitness(ind[_ALGO_GENE], fit[0], bad_threshold=bad_threshold)
    return fit


def _ea_simple_with_bandit(
    population,
    toolbox,
    cxpb: float,
    mutpb: float,
    ngen: int,
    stats,
    halloffame,
    *,
    bandit: MultiArmedBandit | None,
    bandit_bias: float,
    bad_fitness: float,
    ranking: RankingTrickConfig,
):
    """eaSimple with optional bandit updates and ranker preselection."""
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    ranker = PairwiseLinearRanker(random_state=ranking.random_state)
    archive_X: list[list[float]] = []
    archive_y: list[float] = []

    invalid = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid:
        _evaluate_and_update_bandit(ind, toolbox.evaluate, bandit, bad_fitness)
        archive_X.append(list(ind))
        archive_y.append(float(ind.fitness.values[0]))

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid), **record)

    for gen in range(1, ngen + 1):
        if ranking.enabled and len(archive_y) >= ranking.warmup_evals:
            X_arr = np.asarray(archive_X, dtype=float)
            y_arr = np.asarray(archive_y, dtype=float)
            ranker.fit_from_archive(X_arr, y_arr, max_pairs=ranking.max_pairs)

        pool_size = len(population)
        if ranking.enabled and ranking.oversample > 1:
            pool_size = len(population) * int(ranking.oversample)

        offspring = toolbox.select(population, pool_size)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2], strict=False):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        if ranking.enabled and ranker.is_fitted and len(offspring) > len(population):
            cand = np.asarray([list(ind) for ind in offspring], dtype=float)
            scores = ranker.score(cand)
            keep_idx = np.argsort(scores)[-len(population) :]
            offspring = [offspring[int(i)] for i in keep_idx]

        invalid_off = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_off:
            _evaluate_and_update_bandit(ind, toolbox.evaluate, bandit, bad_fitness)
            archive_X.append(list(ind))
            archive_y.append(float(ind.fitness.values[0]))

        population[:] = offspring
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_off), **record)

    return population, logbook


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 15
    population_size: int = 40
    cv_splits: int = 3
    random_state: int = 0
    metric: MetricName = "silhouette"
    cx_prob: float = 0.5
    mut_prob: float = 0.3
    mut_indpb: float = 0.25
    tournament_size: int = 3
    bandit: BanditPolicy = "none"
    bandit_bias: float = 0.5
    bandit_temperature: float = 1.0
    bandit_ucb_c: float = 1.4142135623730951
    ranking_trick: bool = False
    ranking_warmup: int = 30
    ranking_oversample: int = 3
    ranking_max_pairs: int = 2000


@dataclass(frozen=True)
class EvolutionResult:
    best_individual: list[int]
    best_fitness: float
    best_pipeline: object
    bandit_arm_pulls: tuple[int, ...] | None = None
    logbook: tuple[dict[str, object], ...] | None = None


def run_evolution(
    X: np.ndarray,
    y_eval: np.ndarray | None,
    space: SearchSpace,
    config: EvolutionConfig,
) -> EvolutionResult:
    _register_deap_types()
    rng = random.Random(config.random_state)
    random.seed(config.random_state)
    np.random.seed(config.random_state)

    bandit = make_bandit(
        config.bandit,
        N_ALGO_CHOICES,
        rng=rng,
        exploration_c=config.bandit_ucb_c,
        temperature=config.bandit_temperature,
    )
    bad_fitness = -1e9
    ranking = RankingTrickConfig(
        enabled=config.ranking_trick,
        warmup_evals=config.ranking_warmup,
        oversample=config.ranking_oversample,
        max_pairs=config.ranking_max_pairs,
        random_state=config.random_state,
    )

    toolbox = base.Toolbox()

    def _init_genes() -> list[int]:
        genes = [
            rng.randint(0, 1),
            rng.randint(0, 1),
            rng.randint(2, space.max_pca),
            rng.randint(0, N_ALGO_CHOICES - 1),
            rng.randint(2, space.max_k),
            rng.randint(0, 2),
        ]
        if bandit is not None and rng.random() < config.bandit_bias:
            genes[_ALGO_GENE] = bandit.select_arm()
        space.clip(genes)
        return genes

    toolbox.register("individual", tools.initIterate, creator.IndividualCluster, _init_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=config.population_size)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        lambda ind: _mutate(
            ind,
            space,
            config.mut_indpb,
            bandit=bandit,
            bandit_bias=config.bandit_bias,
            rng=rng,
        ),
    )
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)

    def _evaluate(ind):
        return evaluate_individual(
            list(ind),
            X,
            y_eval,
            space,
            metric=config.metric,
            cv_splits=config.cv_splits,
            random_state=config.random_state,
        )

    toolbox.register("evaluate", _evaluate)

    pop = toolbox.population()
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    _, logbook = _ea_simple_with_bandit(
        pop,
        toolbox,
        cxpb=config.cx_prob,
        mutpb=config.mut_prob,
        ngen=config.generations,
        stats=stats,
        halloffame=hof,
        bandit=bandit,
        bandit_bias=config.bandit_bias,
        bad_fitness=bad_fitness,
        ranking=ranking,
    )
    log_records = tuple(dict(rec) for rec in logbook)

    best = list(hof[0])
    space.clip(best)
    fit = float(hof[0].fitness.values[0])
    pipe = individual_to_pipeline(best, space, random_state=config.random_state)
    pulls = tuple(a.pulls for a in bandit.arms) if bandit is not None else None
    return EvolutionResult(
        best_individual=best,
        best_fitness=fit,
        best_pipeline=pipe,
        bandit_arm_pulls=pulls,
        logbook=log_records,
    )
