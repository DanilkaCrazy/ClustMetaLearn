"""Command-line interface: CSV in, evolutionary clustering pipeline out."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from clustmetalearn.tpot_clustering.encoding import SearchSpace
from clustmetalearn.tpot_clustering.evolve import EvolutionConfig, run_evolution
from clustmetalearn.tpot_clustering.fitness import MetricName


def _load_xy(csv_path: str, label_column: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    df = pd.read_csv(csv_path)
    if label_column is not None:
        if label_column not in df.columns:
            raise SystemExit(f"Column {label_column!r} not found in CSV.")
        y_raw = df[label_column]
        X_df = df.drop(columns=[label_column])
    else:
        y_raw = None
        X_df = df

    num_cols = X_df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        raise SystemExit("No numeric feature columns found after removing the label column.")
    X = X_df[num_cols].to_numpy(dtype=float)
    if y_raw is not None:
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw.to_numpy()
        else:
            y = pd.factorize(y_raw)[0].astype(np.int64)
    else:
        y = None

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise SystemExit("Features contain NaN or Inf; clean the CSV or impute before running.")

    return X, y


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evolve a sklearn clustering pipeline from CSV.",
    )
    p.add_argument("csv_path", help="Path to CSV with numeric features.")
    p.add_argument(
        "-l",
        "--label-column",
        default=None,
        help="Optional column name for ground-truth labels (enables external scoring).",
    )
    p.add_argument(
        "--metric",
        choices=[
            "silhouette",
            "silhouette_exact",
            "calinski_harabasz",
            "davies_bouldin",
            "ari",
        ],
        default=None,
        help="Fitness metric (default: silhouette or ari if --label-column).",
    )
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--cv", type=int, default=3, help="KFold splits (>=2).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--bandit",
        choices=["none", "ucb1", "softmax"],
        default="none",
        help="MAB policy for algorithm gene (none, ucb1, softmax).",
    )
    p.add_argument(
        "--bandit-bias",
        type=float,
        default=0.5,
        help="Probability of setting algo gene from bandit on init/mutation (0=uniform).",
    )
    p.add_argument(
        "--bandit-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature (only for --bandit softmax).",
    )
    p.add_argument(
        "--bandit-ucb-c",
        type=float,
        default=1.4142135623730951,
        help="UCB1 exploration constant (only for --bandit ucb1).",
    )
    p.add_argument(
        "--ranking-trick",
        action="store_true",
        help="Pre-rank oversampled candidates with a pairwise ranker.",
    )
    p.add_argument(
        "--ranking-warmup",
        type=int,
        default=30,
        help="Number of evaluated individuals before fitting the ranker.",
    )
    p.add_argument(
        "--ranking-oversample",
        type=int,
        default=3,
        help="Candidate pool multiplier per generation (only used when --ranking-trick).",
    )
    p.add_argument(
        "--ranking-max-pairs",
        type=int,
        default=2000,
        help="Maximum number of random pairs to train the pairwise ranker on.",
    )
    p.add_argument(
        "--output-pipeline",
        default=None,
        help="Optional path to save best pipeline (joblib).",
    )
    p.add_argument(
        "--mlflow",
        action="store_true",
        help="Log run to MLflow.",
    )
    p.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI env or sqlite:///./mlruns/mlflow.db).",
    )
    p.add_argument(
        "--mlflow-experiment",
        default="clustmetalearn-tpot",
        help="MLflow experiment name.",
    )
    p.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name.",
    )
    p.add_argument(
        "--cvisel-models-dir",
        default=None,
        help="Directory with trained CVIsel (models/); sets --metric from meta-model.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    X, y_eval = _load_xy(args.csv_path, args.label_column)

    metric: MetricName
    if args.metric is not None:
        metric = args.metric  # type: ignore[assignment]
    elif args.cvisel_models_dir:
        from clustmetalearn.tpot_clustering.cvisel_bridge import metric_from_cvisel

        metric = metric_from_cvisel(X, Path(args.cvisel_models_dir))
        print("CVIsel metric:", metric)
    elif args.label_column:
        metric = "ari"
    else:
        metric = "silhouette"

    if metric == "ari" and args.label_column is None:
        raise SystemExit("Metric 'ari' requires --label-column.")

    if X.shape[0] < args.cv + 1:
        raise SystemExit(f"Need more rows than CV folds ({args.cv}). Got n_samples={X.shape[0]}.")

    space = SearchSpace.from_shape(X.shape[0], X.shape[1])
    config = EvolutionConfig(
        generations=args.generations,
        population_size=args.population,
        cv_splits=args.cv,
        random_state=args.seed,
        metric=metric,
        bandit=args.bandit,
        bandit_bias=args.bandit_bias,
        bandit_temperature=args.bandit_temperature,
        bandit_ucb_c=args.bandit_ucb_c,
        ranking_trick=args.ranking_trick,
        ranking_warmup=args.ranking_warmup,
        ranking_oversample=args.ranking_oversample,
        ranking_max_pairs=args.ranking_max_pairs,
    )

    result = run_evolution(X, y_eval, space, config)
    print("Best CV score (higher is better):", result.best_fitness)
    print("Best genome:", result.best_individual)
    if result.bandit_arm_pulls is not None:
        print(
            "Bandit arm pulls (KMeans, Agglo, GMM, MiniBatch):",
            result.bandit_arm_pulls,
        )
    print("Best pipeline:")
    print(result.best_pipeline)

    if args.output_pipeline:
        import joblib

        joblib.dump(result.best_pipeline, args.output_pipeline)
        print("Saved:", args.output_pipeline)

    if args.mlflow:
        from clustmetalearn.tpot_clustering.mlflow_tracking import log_evolution_run

        run_id = log_evolution_run(
            result,
            config,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
            run_name=args.mlflow_run_name,
            csv_path=args.csv_path,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            pipeline_path=args.output_pipeline,
        )
        print("MLflow run id:", run_id)


if __name__ == "__main__":
    main(sys.argv[1:])
