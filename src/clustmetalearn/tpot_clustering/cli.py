"""Command-line interface: CSV in, evolutionary clustering pipeline out."""

from __future__ import annotations

import argparse
import sys

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
        description="Evolve a sklearn clustering pipeline from a CSV (ClustMetaLearn / DEAP).",
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
        choices=["silhouette", "calinski_harabasz", "davies_bouldin", "ari"],
        default=None,
        help="Fitness metric. Default: silhouette without labels, ari with --label-column.",
    )
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--cv", type=int, default=3, help="KFold splits (>=2).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-pipeline",
        default=None,
        help="Optional path to save best pipeline (joblib).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    metric: MetricName
    if args.metric is None:
        metric = "ari" if args.label_column else "silhouette"
    else:
        metric = args.metric  # type: ignore[assignment]

    if metric == "ari" and args.label_column is None:
        raise SystemExit("Metric 'ari' requires --label-column.")

    X, y_eval = _load_xy(args.csv_path, args.label_column)
    if X.shape[0] < args.cv + 1:
        raise SystemExit(f"Need more rows than CV folds ({args.cv}). Got n_samples={X.shape[0]}.")

    space = SearchSpace.from_shape(X.shape[0], X.shape[1])
    config = EvolutionConfig(
        generations=args.generations,
        population_size=args.population,
        cv_splits=args.cv,
        random_state=args.seed,
        metric=metric,
    )

    result = run_evolution(X, y_eval, space, config)
    print("Best CV score (higher is better):", result.best_fitness)
    print("Best genome:", result.best_individual)
    print("Best pipeline:")
    print(result.best_pipeline)

    if args.output_pipeline:
        import joblib

        joblib.dump(result.best_pipeline, args.output_pipeline)
        print("Saved:", args.output_pipeline)


if __name__ == "__main__":
    main(sys.argv[1:])
