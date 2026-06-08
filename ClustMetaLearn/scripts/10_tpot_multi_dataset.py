#!/usr/bin/env python3
"""Full TPOT evolution on multiple benchmark datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import adjusted_rand_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from clustmetalearn.tpot_clustering.cvisel_bridge import metric_from_cvisel
from clustmetalearn.tpot_clustering.encoding import SearchSpace
from clustmetalearn.tpot_clustering.evolve import EvolutionConfig, run_evolution

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
OUT_DIR = BASE_DIR / "models" / "tpot_runs"

DATASETS = [
    ("iris", "sklearn"),
    ("wine", "sklearn"),
    ("seeds", "labeled"),
    ("zoo", "labeled"),
    ("ecoli", "labeled"),
]


def _load_sklearn(name: str):
    loaders = {"iris": load_iris, "wine": load_wine}
    b = loaders[name]()
    return b.data.astype(np.float64), b.target.astype(np.int64)


def _load_npy(name: str):
    import urllib.request

    cache = BASE_DIR / "examples" / "benchmark_cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/hj-n/labeled-datasets/master/npy/{name}"
    paths = {}
    for fname in ("data.npy", "label.npy"):
        p = cache / fname
        if not p.exists():
            with urllib.request.urlopen(f"{base}/{fname}", timeout=30) as resp:
                p.write_bytes(resp.read())
        paths[fname] = p
    return np.load(paths["data.npy"]).astype(np.float64), np.load(paths["label.npy"]).astype(np.int64)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, source in DATASETS:
        print(f"[tpot] {name} ({source})", flush=True)
        if source == "sklearn":
            X, y = _load_sklearn(name)
        else:
            X, y = _load_npy(name)
        metric = metric_from_cvisel(X, MODELS_DIR)
        space = SearchSpace.from_shape(X.shape[0], X.shape[1])
        config = EvolutionConfig(
            generations=15,
            population_size=40,
            cv_splits=3,
            random_state=42,
            metric=metric,
            bandit="ucb1",
            bandit_bias=0.5,
            ranking_trick=True,
            ranking_warmup=30,
            ranking_oversample=3,
        )
        result = run_evolution(X, y, space, config)
        fitted = clone(result.best_pipeline)
        fitted.fit(X)
        pred = fitted.predict(X)
        ari = float(adjusted_rand_score(y, pred))
        out_path = OUT_DIR / f"{name}_pipeline.joblib"
        import joblib

        joblib.dump(fitted, out_path)
        row = {
            "dataset": name,
            "metric": metric,
            "cv_fitness": round(result.best_fitness, 4),
            "ari": round(ari, 4),
            "genome": result.best_individual,
            "pipeline": str(out_path),
        }
        rows.append(row)
        print(f"  cv={row['cv_fitness']} ari={row['ari']} -> {out_path}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / "tpot_multi_dataset.csv"
    df.to_csv(csv_path, index=False)
    md = REPORTS_DIR / "tpot_multi_dataset.md"
    md.write_text("# TPOT multi-dataset runs\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"[SUCCESS] {csv_path}")
    print(f"[SUCCESS] {md}")


if __name__ == "__main__":
    main()
