#!/usr/bin/env python3
"""Smoke + integration test for all ClustMetaLearn recommendation modes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from clustmetalearn.meta.labels import best_algorithm_for_dataset
from clustmetalearn.meta.models import load_bundle
from clustmetalearn.meta.recommend import recommend_from_table
from clustmetalearn.meta.features import extract_meta_features_from_csv
from clustmetalearn.meta.hp_intervals import load_hp_intervals, format_hp_guidance
from clustmetalearn.tpot_clustering.encoding import SearchSpace
from clustmetalearn.tpot_clustering.evolve import EvolutionConfig, run_evolution
from clustmetalearn.tpot_clustering.cvisel_bridge import metric_from_cvisel

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"
IRIS_CSV = BASE_DIR / "examples" / "datasets" / "iris" / "iris.csv"
K_MAX = 6


@dataclass
class ModeResult:
    mode: str
    status: str
    details: dict
    elapsed_sec: float


def _run_clustering(X, y, algorithm: str, n_clusters: int, random_state: int = 42):
    import importlib.util

    benchmark = BASE_DIR / "scripts" / "08_algorithm_benchmark_report.py"
    spec = importlib.util.spec_from_file_location("bench08", benchmark)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pred = mod._run_clustering(X, algorithm=algorithm, n_clusters=n_clusters, random_state=random_state)
    if pred is None:
        return np.nan
    return float(adjusted_rand_score(y, pred))


def _subprocess_env() -> dict:
    import os

    env = os.environ.copy()
    src = str(BASE_DIR / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def preflight() -> list[str]:
    errors: list[str] = []
    meta_path = MODELS_DIR / "meta.json"
    if not meta_path.is_file():
        errors.append("missing models/meta.json — run scripts/03_meta_models.py")
        return errors

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_feat = len(meta.get("feature_cols", []))
    bundle = load_bundle(MODELS_DIR)

    if bundle.cvisel is None:
        errors.append("CVIsel not loaded (feature mismatch or missing cvisel.joblib)")
    if bundle.algrank is None:
        errors.append("AlgRank not loaded (feature mismatch or missing algrank.joblib)")

    for name, path in (("cvisel", MODELS_DIR / "cvisel.joblib"), ("algrank", MODELS_DIR / "algrank.joblib")):
        if path.is_file():
            clf = joblib.load(path)
            expected = getattr(clf, "n_features_in_", None)
            if expected is not None and int(expected) != n_feat:
                errors.append(f"{name}: model expects {expected} features, meta.json has {n_feat}")

    if not (MODELS_DIR / "hp_intervals.json").is_file():
        errors.append("missing models/hp_intervals.json")

    if errors:
        return errors

    rec = recommend_from_table(str(IRIS_CSV), MODELS_DIR)
    if rec.predicted_algorithm is None:
        errors.append("recommend_from_table returned no algorithm")
    return errors


def mode1_fast_recommend(X, y) -> ModeResult:
    t0 = time.time()
    rec = recommend_from_table(str(IRIS_CSV), MODELS_DIR)
    algo = rec.predicted_algorithm or "kmeans"
    best_ari = -1.0
    best_k = 2
    for k in range(2, K_MAX + 1):
        ari = _run_clustering(X, y, algo, k)
        if ari > best_ari:
            best_ari = ari
            best_k = k
    ref = best_algorithm_for_dataset(X, y, random_state=42, k_max=K_MAX)
    return ModeResult(
        mode="1. CVIsel + AlgRank (clust-meta recommend)",
        status="ok",
        details={
            "predicted_cvi": rec.predicted_cvi,
            "predicted_algorithm": algo,
            "top2": list(rec.top2_algorithms),
            "top3": list(rec.top3_algorithms),
            "best_k": best_k,
            "ari": round(best_ari, 4),
            "reference_algorithm": ref.algorithm,
            "reference_ari": round(ref.ari, 4),
            "algo_hit": int(algo == ref.algorithm),
        },
        elapsed_sec=round(time.time() - t0, 2),
    )


def _search_algorithms(X, y, algorithms: tuple[str, ...]) -> tuple[dict, float]:
    hp_raw = load_hp_intervals(MODELS_DIR / "hp_intervals.json")
    best_ari = -1.0
    best_cfg: dict = {}
    for algo in algorithms:
        guidance = format_hp_guidance(hp_raw.get(algo))
        k_values = guidance["k_search_values"] if guidance else list(range(2, K_MAX + 1))
        for k in k_values:
            ari = _run_clustering(X, y, algo, k)
            if ari > best_ari:
                best_ari = ari
                best_cfg = {"algorithm": algo, "k": k, "k_search_values": k_values}
    return best_cfg, best_ari


def mode4b_algrank_top2_hp_guidance(X, y) -> ModeResult:
    t0 = time.time()
    rec = recommend_from_table(str(IRIS_CSV), MODELS_DIR)
    best_cfg, best_ari = _search_algorithms(X, y, rec.top2_algorithms)
    ref = best_algorithm_for_dataset(X, y, random_state=42, k_max=K_MAX)
    return ModeResult(
        mode="4b. AlgRank top-2 + hp_guidance (k search)",
        status="ok",
        details={
            "top2": list(rec.top2_algorithms),
            "best_config": best_cfg,
            "ari": round(best_ari, 4),
            "reference_algorithm": ref.algorithm,
            "reference_ari": round(ref.ari, 4),
            "algo_hit": int(best_cfg.get("algorithm") == ref.algorithm),
        },
        elapsed_sec=round(time.time() - t0, 2),
    )


def mode4_algrank_hp_guidance(X, y) -> ModeResult:
    t0 = time.time()
    rec = recommend_from_table(str(IRIS_CSV), MODELS_DIR)
    best_cfg, best_ari = _search_algorithms(X, y, rec.top3_algorithms)
    ref = best_algorithm_for_dataset(X, y, random_state=42, k_max=K_MAX)
    return ModeResult(
        mode="4. AlgRank top-3 + hp_guidance (k search)",
        status="ok",
        details={
            "top2": list(rec.top2_algorithms),
            "top3": list(rec.top3_algorithms),
            "best_config": best_cfg,
            "ari": round(best_ari, 4),
            "reference_algorithm": ref.algorithm,
            "reference_ari": round(ref.ari, 4),
            "algo_hit": int(best_cfg.get("algorithm") == ref.algorithm),
        },
        elapsed_sec=round(time.time() - t0, 2),
    )


def _run_tpot(X, y, *, bandit: str, ranking: bool, label: str) -> ModeResult:
    t0 = time.time()
    metric = metric_from_cvisel(X, MODELS_DIR)
    space = SearchSpace.from_shape(X.shape[0], X.shape[1])
    config = EvolutionConfig(
        generations=5,
        population_size=12,
        cv_splits=3,
        random_state=42,
        metric=metric,
        bandit=bandit,  # type: ignore[arg-type]
        bandit_bias=0.5,
        ranking_trick=ranking,
        ranking_warmup=15,
        ranking_oversample=2,
    )
    from sklearn.base import clone

    result = run_evolution(X, y, space, config)
    fitted = clone(result.best_pipeline)
    fitted.fit(X)
    pred = fitted.predict(X)
    ari = float(adjusted_rand_score(y, pred))
    ref = best_algorithm_for_dataset(X, y, random_state=42, k_max=K_MAX)
    return ModeResult(
        mode=label,
        status="ok",
        details={
            "cvisel_metric": metric,
            "best_genome": result.best_individual,
            "cv_fitness": round(result.best_fitness, 4),
            "ari_with_labels": round(ari, 4),
            "reference_algorithm": ref.algorithm,
            "reference_ari": round(ref.ari, 4),
            "bandit_arm_pulls": result.bandit_arm_pulls,
        },
        elapsed_sec=round(time.time() - t0, 2),
    )


def mode2_tpot_cvisel(X, y) -> ModeResult:
    return _run_tpot(
        X,
        y,
        bandit="none",
        ranking=False,
        label="2. TPOT + CVIsel (metric from meta-model)",
    )


def mode3_tpot_full(X, y) -> ModeResult:
    return _run_tpot(
        X,
        y,
        bandit="ucb1",
        ranking=True,
        label="3. TPOT + CVIsel + bandit + ranking",
    )


def mode5_smbo() -> ModeResult:
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(BASE_DIR / "scripts" / "04_smac_tuning.py")],
        cwd=str(BASE_DIR),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    smbo_path = DATA_DIR / "smbo_results.csv"
    best_row = {}
    if smbo_path.is_file():
        df = pd.read_csv(smbo_path)
        if not df.empty:
            best = df.loc[df["internal_fitness"].idxmax()]
            best_row = {
                "algorithm": best["algorithm"],
                "n_clusters": int(best["n_clusters"]),
                "internal_fitness": round(float(best["internal_fitness"]), 4),
                "surrogate_ari": round(float(best["surrogate_ari"]), 4),
            }
    return ModeResult(
        mode="5. SMBO: AlgRank top-3 + ARI-surrogate",
        status="ok" if proc.returncode == 0 else "fail",
        details={
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout.strip().splitlines()[-3:] if proc.stdout else [],
            "stderr_tail": proc.stderr.strip().splitlines()[-3:] if proc.stderr else [],
            "best_config": best_row,
        },
        elapsed_sec=round(time.time() - t0, 2),
    )


def mode6_benchmark_10() -> ModeResult:
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(BASE_DIR / "scripts" / "08_algorithm_benchmark_report.py")],
        cwd=str(BASE_DIR),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    summary = {}
    csv_path = DATA_DIR / "algorithm_benchmark_10.csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        summary = {
            "exact_hit": f"{int(df['algo_exact_hit'].sum())}/{len(df)}",
            "top2_hit": f"{int(df['algo_top2_hit'].sum())}/{len(df)}",
            "top3_hit": f"{int(df['algo_top3_hit'].sum())}/{len(df)}",
            "mean_delta_ari_top1": round(float(df["ari_gap_vs_best"].mean()), 4),
            "mean_delta_ari_top2": round(float(df["ari_gap_top2_vs_best"].mean()), 4),
            "mean_delta_ari_top3": round(float(df["ari_gap_top3_vs_best"].mean()), 4),
        }
    return ModeResult(
        mode="6. Benchmark 10 datasets (CVIsel + AlgRank)",
        status="ok" if proc.returncode == 0 else "fail",
        details={
            "returncode": proc.returncode,
            "summary": summary,
            "stderr": proc.stderr.strip()[-500:] if proc.stderr else "",
        },
        elapsed_sec=round(time.time() - t0, 2),
    )


def format_report(results: list[ModeResult], preflight_errors: list[str]) -> str:
    from datetime import datetime

    lines = [
        "# Тест всех режимов ClustMetaLearn",
        "",
        f"**Запуск:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Датасет для режимов 1–4: `iris` ({IRIS_CSV})",
        f"Модели: `{MODELS_DIR}`",
        "",
    ]
    if preflight_errors:
        lines += ["## Preflight: ОШИБКИ", ""]
        for e in preflight_errors:
            lines.append(f"- {e}")
        lines.append("")
    else:
        lines += ["## Preflight: OK", ""]

    for r in results:
        lines.append(f"## {r.mode}")
        lines.append(f"- **status:** {r.status}")
        lines.append(f"- **time:** {r.elapsed_sec}s")
        for k, v in r.details.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    lines += [
        "## Сводка режимов",
        "",
        "| Режим | Что тестирует |",
        "|-------|---------------|",
        "| 1 | Быстрый совет: CVIsel + AlgRank |",
        "| 2 | Эволюция пайплайна + CVIsel metric |",
        "| 3 | Эволюция + bandit + ranking trick |",
        "| 4b | AlgRank top-2 + hp_guidance k-search |",
        "| 4 | AlgRank top-3 + hp_guidance k-search |",
        "| 5 | SMBO с суррогатом ARI + AlgRank top-3 |",
        "| 6 | Бенчмарк 10 датасетов |",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import os

    os.environ.setdefault("PYTHONPATH", str(BASE_DIR / "src"))

    from datetime import datetime

    print(f"=== All mode tests started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print("=== Preflight ===", flush=True)
    errors = preflight()
    if errors:
        for e in errors:
            print("FAIL:", e)
        print("\nПереобучите модели:")
        print("  PYTHONPATH=src python scripts/03_meta_models.py")
        return 1
    print("Preflight OK")

    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    results: list[ModeResult] = []
    for fn in (
        mode1_fast_recommend,
        mode4b_algrank_top2_hp_guidance,
        mode4_algrank_hp_guidance,
        mode2_tpot_cvisel,
        mode3_tpot_full,
        mode5_smbo,
        mode6_benchmark_10,
    ):
        print(f"\n=== {fn.__name__} ===")
        try:
            if fn in (mode5_smbo, mode6_benchmark_10):
                r = fn()
            else:
                r = fn(X, y)
            results.append(r)
            print(r.mode, "->", r.status, f"({r.elapsed_sec}s)")
            print(json.dumps(r.details, ensure_ascii=False, indent=2))
        except Exception as exc:
            results.append(
                ModeResult(
                    mode=fn.__name__,
                    status="fail",
                    details={"error": str(exc)},
                    elapsed_sec=0.0,
                )
            )
            print("FAIL:", exc)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "all_modes_test.md"
    report_path.write_text(format_report(results, errors), encoding="utf-8")
    print(f"\n=== Report: {report_path} ===")

    failed = [r for r in results if r.status != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
