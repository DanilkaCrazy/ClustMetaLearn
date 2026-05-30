"""DEAP evolution loop for clustering pipelines."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from deap import algorithms, base, creator, tools

from clustmetalearn.tpot_clustering.encoding import (
    N_ALGO_CHOICES,
    SearchSpace,
    individual_to_pipeline,
)
from clustmetalearn.tpot_clustering.fitness import MetricName, evaluate_individual


def _register_deap_types() -> None:
    if not hasattr(creator, "FitnessMaxCluster"):
        creator.create("FitnessMaxCluster", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualCluster"):
        creator.create("IndividualCluster", list, fitness=creator.FitnessMaxCluster)


def _mutate(individual: list, space: SearchSpace, indpb: float) -> tuple:
    for i in range(len(individual)):
        if random.random() >= indpb:
            continue
        if i in (0, 1):
            individual[i] = random.randint(0, 1)
        elif i == 3:
            individual[i] = random.randint(0, N_ALGO_CHOICES - 1)
        elif i == 2:
            individual[i] = random.randint(2, space.max_pca)
        elif i == 4:
            individual[i] = random.randint(2, space.max_k)
        elif i == 5:
            individual[i] = random.randint(0, 2)
    space.clip(individual)
    return (individual,)


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


@dataclass(frozen=True)
class EvolutionResult:
    best_individual: list[int]
    best_fitness: float
    best_pipeline: object


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

    toolbox = base.Toolbox()

    def _init_genes() -> list[int]:
        return [
            rng.randint(0, 1),
            rng.randint(0, 1),
            rng.randint(2, space.max_pca),
            rng.randint(0, N_ALGO_CHOICES - 1),
            rng.randint(2, space.max_k),
            rng.randint(0, 2),
        ]

    toolbox.register("individual", tools.initIterate, creator.IndividualCluster, _init_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=config.population_size)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        lambda ind: _mutate(ind, space, config.mut_indpb),
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

    pop, _logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=config.cx_prob,
        mutpb=config.mut_prob,
        ngen=config.generations,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    best = list(hof[0])
    space.clip(best)
    fit = float(hof[0].fitness.values[0])
    pipe = individual_to_pipeline(best, space, random_state=config.random_state)
    return EvolutionResult(best_individual=best, best_fitness=fit, best_pipeline=pipe)
