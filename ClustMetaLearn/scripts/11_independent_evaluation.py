#!/usr/bin/env python3
"""Independent and integration evaluation for ClustMetaLearn.

This script keeps the production choice of fitting final models on train+val,
but reports quality on the CLM test split and on cross-dataset resampling
protocols. Raw-matrix ARI checks are limited by sample/dataset caps so the
script remains practical on laptops.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from clustmetalearn.meta.constants import (
    ALGORITHM_NAMES,
    CVI_CLASS_TO_METRIC,
    META_FEATURE_COLS,
    NON_TOPO_FEATURE_COLS,
)
from clustmetalearn.meta.hp_intervals import compute_hp_intervals, load_hp_intervals
from clustmetalearn.meta.labels import best_algorithm_for_dataset
from clustmetalearn.meta.models import (
    ClassifierBackend,
    MetaModelBundle,
    feature_vector_from_dict,
    load_bundle,
    predict_top_k,
)
from clustmetalearn.meta.recommend import recommend_from_features
from clustmetalearn.meta.search import _run_clustering, search_best_clustering
from clustmetalearn.meta.train import prepare_training_table


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
NPY_ROOT = DATA_DIR / "labeled_datasets" / "npy"


@dataclass(frozen=True)
class RawDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    was_subsampled: bool


def _feature_cols(use_topo: bool) -> list[str]:
    return list(META_FEATURE_COLS if use_topo else NON_TOPO_FEATURE_COLS)


def _prepare_df() -> pd.DataFrame:
    meta_csv = DATA_DIR / "meta_features.csv"
    labels_csv = DATA_DIR / "algorithm_labels.csv"
    if not meta_csv.is_file():
        raise FileNotFoundError(f"Missing {meta_csv}. Run scripts/02_isa_adjusted_ivms.py first.")
    return prepare_training_table(meta_csv, labels_csv=labels_csv if labels_csv.is_file() else None)


def _fit_bundle_on_rows(
    train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    use_topo: bool,
    backend: ClassifierBackend,
    random_state: int,
    fit_cvisel: bool = True,
) -> tuple[MetaModelBundle, dict[str, dict]]:
    train_df = train_df.copy()
    train_df["split"] = "train"

    cvisel = None
    cvi_classes: list[str] = []
    cvi_df = train_df.dropna(subset=["target_cvi"]) if "target_cvi" in train_df.columns else train_df.iloc[0:0]
    if fit_cvisel and len(cvi_df) >= 3 and cvi_df["target_cvi"].nunique() >= 2:
        try:
            cvisel = _make_resampling_classifier(random_state=random_state)
            cvisel.fit(cvi_df[feature_cols].to_numpy(dtype=float), cvi_df["target_cvi"].astype(str).to_numpy())
            cvi_classes = list(cvisel.classes_)
        except Exception:
            cvisel = None
            cvi_classes = []

    algrank = None
    algo_classes: list[str] = []
    algo_df = train_df.dropna(subset=["best_algorithm"]) if "best_algorithm" in train_df.columns else train_df.iloc[0:0]
    if len(algo_df) >= 3 and algo_df["best_algorithm"].nunique() >= 2:
        algrank = _make_resampling_classifier(random_state=random_state)
        algrank.fit(algo_df[feature_cols].to_numpy(dtype=float), algo_df["best_algorithm"].astype(str).to_numpy())
        algo_classes = list(algrank.classes_)

    intervals = compute_hp_intervals(algo_df) if not algo_df.empty else {}
    return (
        MetaModelBundle(
            cvisel=cvisel,
            algrank=algrank,
            feature_cols=feature_cols,
            cvi_classes=cvi_classes,
            algo_classes=algo_classes,
            use_topo=use_topo,
            classifier_backend=backend,
        ),
        intervals,
    )


def _make_resampling_classifier(*, random_state: int) -> RandomForestClassifier:
    """Small deterministic classifier for repeated nested/LODO fits."""
    return RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )


def _evaluate_meta_rows(
    bundle: MetaModelBundle,
    df: pd.DataFrame,
    *,
    protocol: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    if bundle.algrank is None or df.empty:
        return pd.DataFrame(rows)

    for row in df.dropna(subset=["best_algorithm"]).itertuples(index=False):
        row_dict = row._asdict()
        Xv = feature_vector_from_dict(row_dict, bundle.feature_cols)
        pred = str(bundle.algrank.predict(Xv)[0])
        top2 = predict_top_k(bundle.algrank, Xv, k=2)[0]
        top3 = predict_top_k(bundle.algrank, Xv, k=3)[0]
        truth = str(row_dict["best_algorithm"])
        rows.append(
            {
                "protocol": protocol,
                "dataset": row_dict["dataset"],
                "split": row_dict.get("split"),
                "true_algorithm": truth,
                "predicted_algorithm": pred,
                "top2_algorithms": ",".join(top2),
                "top3_algorithms": ",".join(top3),
                "top1_hit": int(pred == truth),
                "top2_hit": int(truth in top2),
                "top3_hit": int(truth in top3),
            }
        )
    return pd.DataFrame(rows)


def _baseline_meta_rows(train_df: pd.DataFrame, eval_df: pd.DataFrame, *, protocol: str) -> pd.DataFrame:
    train = train_df.dropna(subset=["best_algorithm"])
    eval_labeled = eval_df.dropna(subset=["best_algorithm"])
    if train.empty or eval_labeled.empty:
        return pd.DataFrame()

    majority = str(train["best_algorithm"].mode().iloc[0])
    rows = []
    for method, algo in [("majority_algorithm", majority), *[(f"always_{a}", a) for a in ALGORITHM_NAMES]]:
        hits = (eval_labeled["best_algorithm"].astype(str) == algo).astype(int)
        rows.append(
            {
                "protocol": protocol,
                "method": method,
                "n_datasets": int(len(eval_labeled)),
                "top1_accuracy": float(hits.mean()),
                "top2_accuracy": np.nan,
                "top3_accuracy": np.nan,
            }
        )

    class_count = max(1, train["best_algorithm"].nunique())
    rows.append(
        {
            "protocol": protocol,
            "method": "random_uniform_expected",
            "n_datasets": int(len(eval_labeled)),
            "top1_accuracy": 1.0 / class_count,
            "top2_accuracy": min(1.0, 2.0 / class_count),
            "top3_accuracy": min(1.0, 3.0 / class_count),
        }
    )
    return pd.DataFrame(rows)


def _summarize_hits(df: pd.DataFrame, *, protocol: str, method: str = "ClustMetaLearn") -> dict:
    if df.empty:
        return {
            "protocol": protocol,
            "method": method,
            "n_datasets": 0,
            "top1_accuracy": np.nan,
            "top2_accuracy": np.nan,
            "top3_accuracy": np.nan,
        }
    return {
        "protocol": protocol,
        "method": method,
        "n_datasets": int(len(df)),
        "top1_accuracy": float(df["top1_hit"].mean()),
        "top2_accuracy": float(df["top2_hit"].mean()),
        "top3_accuracy": float(df["top3_hit"].mean()),
    }


def _splitter_for_labels(y: np.ndarray, *, n_splits: int, random_state: int):
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    if min_count >= 2:
        splits = min(n_splits, min_count)
        return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=random_state)


def _nested_kfold_eval(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    use_topo: bool,
    backend: ClassifierBackend,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    labeled = df.dropna(subset=["best_algorithm"]).reset_index(drop=True)
    if len(labeled) < 4:
        return pd.DataFrame()

    y = labeled["best_algorithm"].astype(str).to_numpy()
    splitter = _splitter_for_labels(y, n_splits=n_splits, random_state=random_state)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(labeled, y), start=1):
        print(f"  nested fold {fold}/{splitter.get_n_splits()}...", flush=True)
        bundle, _ = _fit_bundle_on_rows(
            labeled.iloc[train_idx],
            feature_cols=feature_cols,
            use_topo=use_topo,
            backend=backend,
            random_state=random_state,
            fit_cvisel=False,
        )
        fold_rows = _evaluate_meta_rows(bundle, labeled.iloc[test_idx], protocol=f"nested_{n_splits}fold")
        if not fold_rows.empty:
            fold_rows["fold"] = fold
            rows.append(fold_rows)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _lodo_eval(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    use_topo: bool,
    backend: ClassifierBackend,
    random_state: int,
    max_folds: int | None,
) -> pd.DataFrame:
    labeled = df.dropna(subset=["best_algorithm"]).reset_index(drop=True)
    if len(labeled) < 4:
        return pd.DataFrame()

    rows = []
    total = len(labeled) if max_folds is None else min(max_folds, len(labeled))
    for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(labeled), start=1):
        if max_folds is not None and fold > max_folds:
            break
        if fold == 1 or fold % 10 == 0 or fold == total:
            print(f"  LODO fold {fold}/{total}...", flush=True)
        bundle, _ = _fit_bundle_on_rows(
            labeled.iloc[train_idx],
            feature_cols=feature_cols,
            use_topo=use_topo,
            backend=backend,
            random_state=random_state,
            fit_cvisel=False,
        )
        fold_rows = _evaluate_meta_rows(bundle, labeled.iloc[test_idx], protocol="leave_one_dataset_out")
        if not fold_rows.empty:
            fold_rows["fold"] = fold
            rows.append(fold_rows)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_raw_dataset(name: str, *, max_samples: int, random_state: int) -> RawDataset | None:
    data_path = NPY_ROOT / name / "data.npy"
    label_path = NPY_ROOT / name / "label.npy"
    if not data_path.is_file() or not label_path.is_file():
        return None

    X = np.load(data_path).astype(np.float64)
    y_raw = np.load(label_path, allow_pickle=True)
    y = LabelEncoder().fit_transform(y_raw.astype(str))
    if X.ndim != 2 or len(X) != len(y) or len(np.unique(y)) < 2:
        return None

    was_subsampled = False
    if max_samples > 0 and len(y) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        was_subsampled = True
    return RawDataset(name=name, X=X, y=y, was_subsampled=was_subsampled)


def _ari_for_config(X: np.ndarray, y: np.ndarray, *, algorithm: str, n_clusters: int) -> float:
    labels = _run_clustering(X, algorithm=algorithm, n_clusters=n_clusters)
    if labels is None:
        return np.nan
    return float(adjusted_rand_score(y, labels))


def _evaluate_candidate_set(
    X: np.ndarray,
    y: np.ndarray,
    *,
    algorithms: Iterable[str],
    metric: str,
    hp_intervals: dict[str, dict],
    method: str,
    oracle_ari: float,
) -> dict:
    algorithms = list(dict.fromkeys(a for a in algorithms if a in ALGORITHM_NAMES))
    result = search_best_clustering(X, algorithms, metric=metric, hp_intervals=hp_intervals)
    if result is None:
        return {
            "method": method,
            "selected_algorithm": None,
            "selected_k": None,
            "internal_score": np.nan,
            "ari": np.nan,
            "delta_ari": np.nan,
            "relative_ari": np.nan,
        }
    ari = _ari_for_config(X, y, algorithm=result.algorithm, n_clusters=result.n_clusters)
    relative = ari / oracle_ari if np.isfinite(ari) and oracle_ari > 0 else np.nan
    return {
        "method": method,
        "selected_algorithm": result.algorithm,
        "selected_k": result.n_clusters,
        "internal_score": result.internal_score,
        "ari": ari,
        "delta_ari": ari - oracle_ari if np.isfinite(ari) else np.nan,
        "relative_ari": relative,
    }


def _raw_pipeline_eval(
    df: pd.DataFrame,
    bundle: MetaModelBundle,
    *,
    hp_intervals: dict[str, dict],
    split_name: str,
    max_raw_datasets: int,
    max_samples: int,
    random_state: int,
    k_max: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    cvi_rows: list[dict] = []
    candidates = df[(df["split"] == split_name) & df["best_algorithm"].notna()].copy()

    processed = 0
    for row in candidates.itertuples(index=False):
        if max_raw_datasets > 0 and processed >= max_raw_datasets:
            break
        row_dict = row._asdict()
        raw = _load_raw_dataset(str(row_dict["dataset"]), max_samples=max_samples, random_state=random_state)
        if raw is None:
            continue

        processed += 1
        print(
            f"  raw {processed}/{max_raw_datasets or 'all'}: {raw.name} "
            f"n={raw.X.shape[0]} d={raw.X.shape[1]}...",
            flush=True,
        )
        feats = {c: float(row_dict[c]) for c in bundle.feature_cols}
        oracle = best_algorithm_for_dataset(raw.X, raw.y, random_state=random_state, k_max=k_max)
        oracle_ari = float(oracle.ari)

        rec_none = recommend_from_features(feats, bundle, hp_intervals=hp_intervals, search_mode="none")
        predicted_metric = rec_none.predicted_cvi_metric or "silhouette"
        predicted_algorithm = rec_none.predicted_algorithm or "kmeans"
        top2 = list(rec_none.top2_algorithms) or [predicted_algorithm]
        top3 = list(rec_none.top3_algorithms) or top2

        method_specs = [
            ("meta_top1_internal", [predicted_algorithm]),
            ("meta_top2_internal", top2),
            ("meta_top3_internal", top3),
            ("all_algorithms_internal", list(ALGORITHM_NAMES)),
            *[(f"always_{algo}", [algo]) for algo in ALGORITHM_NAMES],
        ]
        for method, algorithms in method_specs:
            out = _evaluate_candidate_set(
                raw.X,
                raw.y,
                algorithms=algorithms,
                metric=predicted_metric,
                hp_intervals=hp_intervals,
                method=method,
                oracle_ari=oracle_ari,
            )
            out.update(
                {
                    "protocol": f"{split_name}_raw_full_pipeline",
                    "dataset": raw.name,
                    "n_samples_eval": int(raw.X.shape[0]),
                    "n_features": int(raw.X.shape[1]),
                    "was_subsampled": int(raw.was_subsampled),
                    "predicted_cvi": rec_none.predicted_cvi,
                    "predicted_metric": predicted_metric,
                    "predicted_algorithm": predicted_algorithm,
                    "top2_algorithms": ",".join(top2),
                    "top3_algorithms": ",".join(top3),
                    "oracle_algorithm": oracle.algorithm,
                    "oracle_k": oracle.n_clusters,
                    "oracle_ari": oracle_ari,
                }
            )
            rows.append(out)

        cvi_metrics = {
            "predicted_cvi": predicted_metric,
            "silhouette": "silhouette",
            "calinski_harabasz": "calinski_harabasz",
            "davies_bouldin": "davies_bouldin",
        }
        for cvi_name, metric in cvi_metrics.items():
            out = _evaluate_candidate_set(
                raw.X,
                raw.y,
                algorithms=ALGORITHM_NAMES,
                metric=metric,
                hp_intervals=hp_intervals,
                method=f"cvi_{cvi_name}",
                oracle_ari=oracle_ari,
            )
            cvi_rows.append(
                {
                    "dataset": raw.name,
                    "cvi_policy": cvi_name,
                    "metric": metric,
                    "selected_algorithm": out["selected_algorithm"],
                    "selected_k": out["selected_k"],
                    "oracle_algorithm": oracle.algorithm,
                    "algorithm_hit": int(out["selected_algorithm"] == oracle.algorithm),
                    "ari": out["ari"],
                    "oracle_ari": oracle_ari,
                    "delta_ari": out["delta_ari"],
                    "was_subsampled": int(raw.was_subsampled),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(cvi_rows)


def _format_float(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["_Нет данных._"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines


def _load_existing_csv(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def _write_report(
    *,
    independent_summary: pd.DataFrame,
    integration_summary: pd.DataFrame,
    nested_summary: pd.DataFrame,
    lodo_summary: pd.DataFrame,
    raw_summary: pd.DataFrame,
    cvi_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    max_samples: int,
) -> str:
    lines: list[str] = [
        "# Независимая и интеграционная оценка ClustMetaLearn",
        "",
        "## Что проверяется",
        "",
        "- `test_meta_holdout` — финальная модель обучена на train+val, качество считается только на CLM test.",
        "- `train_val_integration` — интеграционный in-sample контроль на train+val, не является независимой оценкой.",
        "- `nested_kfold` — переобучение AlgRank на внешних фолдах; каждая строка оценивается моделью, которая её не видела.",
        "- `leave_one_dataset_out` — максимально строгий вариант: один датасет исключается из обучения и становится тестом.",
        "- `test_raw_full_pipeline` — сквозной ARI на raw `.npy`: CVIsel -> AlgRank top-k -> поиск k по внутренней CVI -> ARI.",
        "",
        f"Raw-оценка использует deterministic subsampling до `{max_samples}` объектов, если датасет больше этого лимита.",
        "",
        "## Algorithm Hit Метрики",
        "",
    ]

    hit_tables = [
        independent_summary,
        integration_summary,
        nested_summary,
        lodo_summary,
        baseline_summary,
    ]
    hit_df = pd.concat([t for t in hit_tables if not t.empty], ignore_index=True) if any(not t.empty for t in hit_tables) else pd.DataFrame()
    if not hit_df.empty:
        view = hit_df.copy()
        for col in ("top1_accuracy", "top2_accuracy", "top3_accuracy"):
            view[col] = view[col].map(lambda x: _format_float(float(x)) if not pd.isna(x) else "n/a")
        lines.extend(_markdown_table(view, ["protocol", "method", "n_datasets", "top1_accuracy", "top2_accuracy", "top3_accuracy"]))
    else:
        lines.append("_Нет meta-level результатов._")

    lines.extend(["", "## Сквозная ARI Оценка На Test", ""])
    if not raw_summary.empty:
        raw_view = raw_summary.copy()
        for col in ("mean_ari", "mean_delta_ari", "mean_relative_ari"):
            raw_view[col] = raw_view[col].map(lambda x: _format_float(float(x)) if not pd.isna(x) else "n/a")
        lines.extend(_markdown_table(raw_view, ["method", "n_datasets", "mean_ari", "mean_delta_ari", "mean_relative_ari"]))
    else:
        lines.append("_Raw `.npy` датасеты для test не найдены или были пропущены._")

    lines.extend(["", "## CVI Alignment", ""])
    if not cvi_summary.empty:
        cvi_view = cvi_summary.copy()
        for col in ("algorithm_hit_rate", "mean_ari", "mean_delta_ari"):
            cvi_view[col] = cvi_view[col].map(lambda x: _format_float(float(x)) if not pd.isna(x) else "n/a")
        lines.extend(_markdown_table(cvi_view, ["cvi_policy", "n_datasets", "algorithm_hit_rate", "mean_ari", "mean_delta_ari"]))
    else:
        lines.append("_CVI alignment не посчитан._")

    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "- Для качества выбора алгоритма ориентироваться на `test_meta_holdout`, `nested_kfold` и `leave_one_dataset_out`.",
            "- Для пользовательского качества ориентироваться на `test_raw_full_pipeline`, потому что там оценивается весь путь до ARI.",
            "- `train_val_integration` нужен только как проверка, что интегрированный пайплайн работает на данных, похожих на обучающие.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["rf", "gbm"], default="gbm")
    parser.add_argument("--use-topo", action="store_true", default=False)
    parser.add_argument("--nested-folds", type=int, default=5)
    parser.add_argument("--max-lodo-folds", type=int, default=0, help="0 means all LODO folds.")
    parser.add_argument("--max-raw-datasets", type=int, default=12, help="0 means all loadable test datasets.")
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw matrix ARI and CVI alignment.")
    parser.add_argument("--skip-resampling", action="store_true", help="Skip nested k-fold and LODO.")
    parser.add_argument("--skip-integration", action="store_true", help="Skip train+val integration table.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = _prepare_df()
    feature_cols = _feature_cols(args.use_topo)

    print("[eval] loading final train+val model artifacts...", flush=True)
    final_bundle = load_bundle(MODELS_DIR)
    hp_path = MODELS_DIR / "hp_intervals.json"
    hp_intervals = load_hp_intervals(hp_path) if hp_path.is_file() else {}

    train_val_df = df[df["split"].isin(("train", "val"))].copy()
    test_df = df[df["split"] == "test"].copy()

    print("[eval] test meta holdout...", flush=True)
    independent_rows = _evaluate_meta_rows(final_bundle, test_df, protocol="test_meta_holdout")
    if args.skip_integration:
        print("[eval] skipping train+val integration.", flush=True)
        integration_rows = pd.DataFrame()
    else:
        print("[eval] train+val integration...", flush=True)
        integration_rows = _evaluate_meta_rows(final_bundle, train_val_df, protocol="train_val_integration")
    if args.skip_resampling:
        print("[eval] skipping nested k-fold and LODO.", flush=True)
        nested_rows = pd.DataFrame()
        lodo_rows = pd.DataFrame()
    else:
        print("[eval] nested k-fold...", flush=True)
        nested_rows = _nested_kfold_eval(
            df,
            feature_cols=feature_cols,
            use_topo=args.use_topo,
            backend=args.backend,
            n_splits=args.nested_folds,
            random_state=args.random_state,
        )
        print("[eval] leave-one-dataset-out...", flush=True)
        lodo_rows = _lodo_eval(
            df,
            feature_cols=feature_cols,
            use_topo=args.use_topo,
            backend=args.backend,
            random_state=args.random_state,
            max_folds=None if args.max_lodo_folds == 0 else args.max_lodo_folds,
        )

    baseline_test = _baseline_meta_rows(train_val_df, test_df, protocol="test_meta_holdout")
    if args.skip_raw:
        print("[eval] skipping raw full-pipeline ARI.", flush=True)
        raw_rows = pd.DataFrame()
        cvi_rows = pd.DataFrame()
    else:
        print("[eval] raw full-pipeline ARI and CVI alignment...", flush=True)
        raw_rows, cvi_rows = _raw_pipeline_eval(
            df,
            final_bundle,
            hp_intervals=hp_intervals,
            split_name="test",
            max_raw_datasets=args.max_raw_datasets,
            max_samples=args.max_samples,
            random_state=args.random_state,
            k_max=args.k_max,
        )

    summaries = {
        "independent": pd.DataFrame([_summarize_hits(independent_rows, protocol="test_meta_holdout")]),
        "integration": pd.DataFrame([_summarize_hits(integration_rows, protocol="train_val_integration")]),
        "nested": pd.DataFrame([_summarize_hits(nested_rows, protocol=f"nested_{args.nested_folds}fold")]),
        "lodo": pd.DataFrame([_summarize_hits(lodo_rows, protocol="leave_one_dataset_out")]),
    }

    raw_summary = (
        raw_rows.groupby("method", as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_ari=("ari", "mean"),
            mean_delta_ari=("delta_ari", "mean"),
            mean_relative_ari=("relative_ari", "mean"),
        )
        .sort_values("mean_delta_ari", ascending=False)
        if not raw_rows.empty
        else pd.DataFrame()
    )
    cvi_summary = (
        cvi_rows.groupby("cvi_policy", as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            algorithm_hit_rate=("algorithm_hit", "mean"),
            mean_ari=("ari", "mean"),
            mean_delta_ari=("delta_ari", "mean"),
        )
        .sort_values("mean_delta_ari", ascending=False)
        if not cvi_rows.empty
        else pd.DataFrame()
    )
    if args.skip_raw:
        raw_summary = _load_existing_csv(DATA_DIR / "full_pipeline_test_ari_summary.csv")
        cvi_summary = _load_existing_csv(DATA_DIR / "cvi_alignment_test_summary.csv")

    outputs = {
        "independent_eval_meta.csv": independent_rows,
        "integration_eval_meta.csv": integration_rows,
        "nested_eval_meta.csv": nested_rows,
        "lodo_eval_meta.csv": lodo_rows,
        "baseline_eval_meta.csv": baseline_test,
        "full_pipeline_test_ari.csv": raw_rows,
        "cvi_alignment_test.csv": cvi_rows,
        "full_pipeline_test_ari_summary.csv": raw_summary,
        "cvi_alignment_test_summary.csv": cvi_summary,
    }
    for filename, table in outputs.items():
        output_path = DATA_DIR / filename
        if table.empty and output_path.is_file():
            continue
        table.to_csv(output_path, index=False)

    report = _write_report(
        independent_summary=summaries["independent"],
        integration_summary=summaries["integration"],
        nested_summary=summaries["nested"],
        lodo_summary=summaries["lodo"],
        raw_summary=raw_summary,
        cvi_summary=cvi_summary,
        baseline_summary=baseline_test,
        max_samples=args.max_samples,
    )
    report_path = REPORTS_DIR / "independent_evaluation.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[Шаг 11 SUCCESS] Report: {report_path}")


if __name__ == "__main__":
    main()
