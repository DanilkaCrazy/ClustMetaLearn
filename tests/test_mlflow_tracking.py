from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clustmetalearn.tpot_clustering.evolve import EvolutionConfig, EvolutionResult
from clustmetalearn.tpot_clustering.mlflow_tracking import (
    evolution_config_to_params,
    log_evolution_run,
)

mlflow = pytest.importorskip("mlflow")


def test_evolution_config_to_params():
    cfg = EvolutionConfig(generations=3, population_size=10, metric="silhouette")
    params = evolution_config_to_params(cfg)
    assert params["generations"] == "3"
    assert params["metric"] == "silhouette"


def test_log_evolution_run_sqlite_store(tmp_path: Path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result = EvolutionResult(
        best_individual=[1, 0, 3, 0, 4, 0],
        best_fitness=0.42,
        best_pipeline=Pipeline([("scaler", StandardScaler())]),
        bandit_arm_pulls=(5, 2, 1, 0),
        logbook=(
            {"gen": 0, "nevals": 10, "max": 0.3, "avg": 0.1},
            {"gen": 1, "nevals": 8, "max": 0.42, "avg": 0.2},
        ),
    )
    config = EvolutionConfig(generations=2, population_size=10)

    run_id = log_evolution_run(
        result,
        config,
        tracking_uri=uri,
        experiment_name="test-exp",
        run_name="unit-test",
        n_samples=100,
        n_features=5,
    )
    assert run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.data.metrics["best_fitness"] == pytest.approx(0.42)
    assert run.data.params["generations"] == "2"

    exp = client.get_experiment_by_name("test-exp")
    assert exp is not None
