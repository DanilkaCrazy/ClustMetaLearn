"""MLflow experiment logging for evolutionary clustering runs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from clustmetalearn.tpot_clustering.evolve import EvolutionConfig, EvolutionResult


def _require_mlflow():
    try:
        import mlflow
    except ImportError as e:
        raise ImportError(
            "MLflow is not installed. Install with: pip install 'clustmetalearn[mlflow]'"
        ) from e
    return mlflow


def evolution_config_to_params(config: EvolutionConfig) -> dict[str, Any]:
    """Serialize EvolutionConfig as MLflow string params."""
    raw = asdict(config) if is_dataclass(config) else dict(config)
    return {k: str(v) for k, v in raw.items()}


def log_evolution_run(
    result: EvolutionResult,
    config: EvolutionConfig,
    *,
    tracking_uri: str | None = None,
    experiment_name: str = "clustmetalearn-tpot",
    run_name: str | None = None,
    csv_path: str | None = None,
    n_samples: int | None = None,
    n_features: int | None = None,
    pipeline_path: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log evolution run to MLflow; return run id."""
    mlflow = _require_mlflow()

    uri = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI",
        "sqlite:///./mlruns/mlflow.db",
    )
    if uri.startswith("sqlite:///"):
        db_path = Path(uri.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)

        params = evolution_config_to_params(config)
        if csv_path:
            params["csv_path"] = csv_path
        if n_samples is not None:
            params["n_samples"] = str(n_samples)
        if n_features is not None:
            params["n_features"] = str(n_features)
        mlflow.log_params(params)

        mlflow.log_metric("best_fitness", float(result.best_fitness))

        if result.logbook:
            for row in result.logbook:
                gen = int(row.get("gen", 0))
                if "max" in row:
                    mlflow.log_metric("gen_max_fitness", float(row["max"]), step=gen)
                if "avg" in row:
                    mlflow.log_metric("gen_avg_fitness", float(row["avg"]), step=gen)
                if "nevals" in row:
                    mlflow.log_metric("gen_nevals", float(row["nevals"]), step=gen)

        if result.bandit_arm_pulls is not None:
            names = ("bandit_kmeans", "bandit_agglo", "bandit_gmm", "bandit_minibatch")
            for name, pulls in zip(names, result.bandit_arm_pulls, strict=False):
                mlflow.log_metric(name, float(pulls))

        genome_payload = {
            "best_genome": result.best_individual,
            "genome_layout": [
                "use_scaler",
                "use_pca",
                "pca_n_components",
                "algorithm",
                "n_clusters",
                "linkage_index",
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            genome_file = tmp_path / "best_genome.json"
            genome_file.write_text(json.dumps(genome_payload, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(genome_file), artifact_path="evolution")

            if pipeline_path and Path(pipeline_path).is_file():
                mlflow.log_artifact(pipeline_path, artifact_path="model")
            else:
                try:
                    import joblib

                    pipe_file = tmp_path / "best_pipeline.joblib"
                    joblib.dump(result.best_pipeline, pipe_file)
                    mlflow.log_artifact(str(pipe_file), artifact_path="model")
                except Exception:
                    pass

        return run.info.run_id
