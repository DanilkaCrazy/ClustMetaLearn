#!/usr/bin/env python3
"""Benchmark meta recommendations on 10 tabular datasets."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from clustmetalearn.meta.features import extract_meta_features_from_csv
from clustmetalearn.meta.labels import best_algorithm_for_dataset
from clustmetalearn.meta.models import feature_vector_from_dict, load_bundle
from clustmetalearn.meta.recommend import recommend_from_table

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
NPY_CACHE = BASE_DIR / "examples" / "benchmark_cache"

BENCHMARK_DATASETS = [
    ("iris", "sklearn"),
    ("wine", "sklearn"),
    ("breast_cancer", "sklearn"),
    ("seeds", "labeled"),
    ("banknote_authentication", "labeled"),
    ("ecoli", "labeled"),
    ("glass_identification", "labeled"),
    ("zoo", "labeled"),
    ("yeast", "labeled"),
    ("mammographic_mass", "labeled"),
]

K_MAX = 6


def _load_sklearn(name: str) -> tuple[np.ndarray, np.ndarray]:
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
    }
    bundle = loaders[name]()
    return bundle.data.astype(np.float64), bundle.target.astype(np.int64)


def _download_npy(name: str) -> tuple[np.ndarray, np.ndarray]:
    import urllib.error
    import urllib.request

    cache_dir = NPY_CACHE / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "data.npy"
    label_path = cache_dir / "label.npy"
    base = f"https://raw.githubusercontent.com/hj-n/labeled-datasets/master/npy/{name}"
    for path, url in ((data_path, f"{base}/data.npy"), (label_path, f"{base}/label.npy")):
        if not path.exists():
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    path.write_bytes(resp.read())
            except (urllib.error.URLError, TimeoutError) as exc:
                raise FileNotFoundError(f"Failed to download {url}: {exc}") from exc
    return np.load(data_path).astype(np.float64), np.load(label_path).astype(np.int64)


def _load_dataset(name: str, source: str) -> tuple[np.ndarray, np.ndarray]:
    if source == "sklearn":
        return _load_sklearn(name)
    return _download_npy(name)


def _run_clustering(
    X: np.ndarray,
    *,
    algorithm: str,
    n_clusters: int,
    use_scaler: bool = True,
    linkage: str = "ward",
    random_state: int = 42,
) -> np.ndarray | None:
    X_work = StandardScaler().fit_transform(X) if use_scaler else X.astype(np.float64)
    try:
        if algorithm == "kmeans":
            return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(X_work)
        if algorithm == "minibatch_kmeans":
            return MiniBatchKMeans(
                n_clusters=n_clusters,
                n_init=3,
                random_state=random_state,
            ).fit_predict(X_work)
        if algorithm == "gmm":
            return GaussianMixture(
                n_components=n_clusters,
                random_state=random_state,
                n_init=2,
            ).fit_predict(X_work)
        if algorithm == "agglomerative":
            return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(X_work)
    except Exception:
        return None
    return None


def _best_among_algorithms(
    X: np.ndarray,
    y: np.ndarray,
    algorithms: list[str],
    *,
    random_state: int = 42,
    k_max: int = K_MAX,
) -> tuple[str, int, float]:
    best_algo = algorithms[0] if algorithms else "kmeans"
    best_k = 2
    best_ari = -1.0
    for algorithm in algorithms:
        k, ari = _best_k_for_algorithm(X, y, algorithm, random_state=random_state, k_max=k_max)
        if ari > best_ari:
            best_algo = algorithm
            best_k = k
            best_ari = ari
    return best_algo, best_k, best_ari


def _best_k_for_algorithm(
    X: np.ndarray,
    y: np.ndarray,
    algorithm: str,
    *,
    random_state: int = 42,
    k_max: int = K_MAX,
) -> tuple[int, float]:
    best_k = 2
    best_ari = -1.0
    n_classes = len(np.unique(y))
    upper = min(k_max, max(n_classes + 2, 3))
    for k in range(2, upper + 1):
        pred = _run_clustering(X, algorithm=algorithm, n_clusters=k, random_state=random_state)
        if pred is None:
            continue
        ari = float(adjusted_rand_score(y, pred))
        if ari > best_ari:
            best_k = k
            best_ari = ari
    return best_k, best_ari


def _evaluate_dataset(name: str, source: str, models_dir: Path) -> dict:
    X, y = _load_dataset(name, source)
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    cache_dir = NPY_CACHE / "csv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"{name}.csv"
    pd.DataFrame(X).to_csv(csv_path, index=False)

    best = best_algorithm_for_dataset(X, y, random_state=42, k_max=K_MAX)
    rec = recommend_from_table(str(csv_path), models_dir, search_mode="top2")

    predicted_algo = rec.predicted_algorithm or "kmeans"
    searched_algo = rec.searched_algorithm or predicted_algo
    top2 = list(rec.top2_algorithms)
    top3 = list(rec.top3_algorithms)
    hp_k_values = (
        list(rec.hp_intervals.get("k_search_values", []))
        if rec.hp_intervals
        else list(range(2, min(7, n_classes + 3)))
    )

    pred_k, pred_algo_best_ari = _best_k_for_algorithm(X, y, predicted_algo)
    searched_k, searched_ari = _best_k_for_algorithm(X, y, searched_algo)
    top2_algo, top2_k, top2_best_ari = _best_among_algorithms(
        X, y, top2 if top2 else [predicted_algo]
    )
    top3_algo, top3_k, top3_best_ari = _best_among_algorithms(
        X, y, top3 if top3 else [predicted_algo]
    )
    pred_at_true_k = _run_clustering(X, algorithm=predicted_algo, n_clusters=n_classes)
    ari_hp_search = -1.0
    for k in hp_k_values:
        pred_hp = _run_clustering(X, algorithm=predicted_algo, n_clusters=k)
        if pred_hp is None:
            continue
        ari_hp_search = max(ari_hp_search, float(adjusted_rand_score(y, pred_hp)))

    pred_at_true_k = _run_clustering(X, algorithm=predicted_algo, n_clusters=n_classes)
    ari_true_k = float(adjusted_rand_score(y, pred_at_true_k)) if pred_at_true_k is not None else np.nan
    ari_hp_k = ari_hp_search if ari_hp_search >= 0 else np.nan

    bundle = load_bundle(models_dir)
    feats = extract_meta_features_from_csv(str(csv_path), include_topology=bundle.use_topo)
    vec = feature_vector_from_dict(feats, bundle.feature_cols)
    surrogate_ari = np.nan
    surrogate_path = models_dir / "ari_surrogate.pkl"
    feature_file = models_dir / "feature_cols.txt"
    if surrogate_path.is_file() and feature_file.is_file():
        sur_features = feature_file.read_text(encoding="utf-8").strip().splitlines()
        sur_vec = feature_vector_from_dict(feats, sur_features)
        surrogate_ari = float(joblib.load(surrogate_path).predict(sur_vec)[0])

    return {
        "dataset": name,
        "source": source,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "best_algorithm": best.algorithm,
        "best_k": best.n_clusters,
        "best_ari": round(best.ari, 4),
        "predicted_cvi": rec.predicted_cvi,
        "predicted_algorithm": predicted_algo,
        "searched_algorithm": searched_algo,
        "searched_n_clusters": rec.searched_n_clusters,
        "searched_internal_score": rec.searched_internal_score,
        "top2_algorithms": ",".join(top2),
        "top3_algorithms": ",".join(top3),
        "algo_exact_hit": int(predicted_algo == best.algorithm),
        "algo_top2_hit": int(best.algorithm in top2),
        "algo_top3_hit": int(best.algorithm in top3),
        "predicted_algo_best_k": pred_k,
        "predicted_algo_best_ari": round(pred_algo_best_ari, 4),
        "searched_algo_best_k": searched_k,
        "searched_algo_best_ari": round(searched_ari, 4),
        "ari_gap_searched_vs_best": round(searched_ari - best.ari, 4),
        "top2_best_algorithm": top2_algo,
        "top2_best_k": top2_k,
        "top2_best_ari": round(top2_best_ari, 4),
        "top3_best_algorithm": top3_algo,
        "top3_best_k": top3_k,
        "top3_best_ari": round(top3_best_ari, 4),
        "ari_gap_top2_vs_best": round(top2_best_ari - best.ari, 4),
        "ari_gap_top3_vs_best": round(top3_best_ari - best.ari, 4),
        "ari_at_true_k": round(ari_true_k, 4),
        "ari_at_hp_k": round(ari_hp_k, 4),
        "surrogate_ari": round(surrogate_ari, 4) if not np.isnan(surrogate_ari) else np.nan,
        "ari_gap_vs_best": round(pred_algo_best_ari - best.ari, 4),
    }


def _format_report(df: pd.DataFrame) -> str:
    n = len(df)
    algo_hits = int(df["algo_exact_hit"].sum())
    top2_hits = int(df["algo_top2_hit"].sum())
    top3_hits = int(df["algo_top3_hit"].sum())
    mean_best = float(df["best_ari"].mean())
    mean_pred = float(df["predicted_algo_best_ari"].mean())
    mean_gap = float(df["ari_gap_vs_best"].mean())
    mean_gap_top2 = float(df["ari_gap_top2_vs_best"].mean())
    mean_gap_top3 = float(df["ari_gap_top3_vs_best"].mean())

    lines = [
        "# Сводный отчёт: 10 датасетов",
        "",
        "Оценка работы meta-пайплайна ClustMetaLearn на 10 табличных датасетах.",
        "Для каждого датасета мета-модели получают только признаки (без меток классов).",
        "Эталон — лучший алгоритм и ARI по полному grid search проекта.",
        "",
        "## Сводка",
        "",
        f"- Датасетов: **{n}**",
        f"- Точное попадание (top-1): **{algo_hits}/{n}** ({100 * algo_hits / n:.1f}%)",
        f"- Попадание в top-2: **{top2_hits}/{n}** ({100 * top2_hits / n:.1f}%)",
        f"- Попадание в top-3: **{top3_hits}/{n}** ({100 * top3_hits / n:.1f}%)",
        f"- Средний лучший ARI (эталон): **{mean_best:.3f}**",
        f"- Средний ARI top-1 + k: **{mean_pred:.3f}** (Δ **{mean_gap:+.3f}**)",
        f"- Средний ARI **top-2 search (default)**: **{float(df['searched_algo_best_ari'].mean()):.3f}** "
        f"(Δ **{float(df['ari_gap_searched_vs_best'].mean()):+.3f}**)",
        f"- Средний ARI top-2 + k: **{float(df['top2_best_ari'].mean()):.3f}** (Δ **{mean_gap_top2:+.3f}**)",
        f"- Средний ARI top-3 + k: **{float(df['top3_best_ari'].mean()):.3f}** (Δ **{mean_gap_top3:+.3f}**)",
        "",
        "## Таблица по датасетам",
        "",
        "| Датасет | Эталон | top-1 | top-2 | top-3 | top-2 hit | top-3 hit | ARI эталон | ARI top-1 | ARI top-2 | ARI top-3 |",
        "|---------|--------|-------|-------|-------|-----------|-----------|------------|-----------|-----------|-----------|",
    ]

    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.best_algorithm} (k={row.best_k}) | "
            f"{row.predicted_algorithm} | {row.top2_algorithms} | {row.top3_algorithms} | "
            f"{'да' if row.algo_top2_hit else 'нет'} | {'да' if row.algo_top3_hit else 'нет'} | "
            f"{row.best_ari:.3f} | {row.predicted_algo_best_ari:.3f} | "
            f"{row.top2_best_ari:.3f} | {row.top3_best_ari:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Дополнительно",
            "",
            "- `ari_at_true_k` — ARI рекомендованного алгоритма при k = числу классов.",
            "- `ari_at_hp_k` — ARI при k из `hp_intervals.k_median`.",
            "- `surrogate_ari` — прогноз ARI суррогатной модели.",
            "",
            "## Выводы",
            "",
        ]
    )

    if algo_hits >= n // 2:
        lines.append("- AlgRank часто угадывает лучшее семейство алгоритма.")
    else:
        lines.append("- AlgRank угадывает точный алгоритм реже, чем в половине случаев.")

    if top3_hits >= int(0.7 * n):
        lines.append("- Top-3 рекомендации покрывают большинство эталонных алгоритмов.")
    else:
        lines.append("- Top-3 рекомендации покрывают эталон нестабильно.")

    if mean_gap > -0.1:
        lines.append("- Рекомендованный алгоритм с подбором k близок к эталону по ARI.")
    else:
        lines.append("- После выбора алгоритма качество заметно ниже эталона; критичен подбор k и scaler.")

    improved = df[df["ari_at_hp_k"] >= df["predicted_algo_best_ari"] - 0.01]
    if len(improved) > 0:
        lines.append(
            f"- На {len(improved)} датасетах перебор k в hp_guidance даёт ARI не хуже подбора по одному алгоритму."
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print("[Шаг 8] Бенчмарк рекомендаций на 10 датасетах...", flush=True)
    rows = []
    for name, source in BENCHMARK_DATASETS:
        print(f"  -> {name} ({source})", flush=True)
        rows.append(_evaluate_dataset(name, source, MODELS_DIR))

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "algorithm_benchmark_10.csv"
    report_path = REPORTS_DIR / "algorithm_benchmark_10.md"
    df.to_csv(csv_path, index=False)
    report_path.write_text(_format_report(df), encoding="utf-8")

    print("\n" + _format_report(df))
    print(f"\n[Шаг 8 SUCCESS] CSV: {csv_path}")
    print(f"[Шаг 8 SUCCESS] Report: {report_path}")


if __name__ == "__main__":
    main()
