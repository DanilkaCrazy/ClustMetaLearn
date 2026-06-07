#!/usr/bin/env python3
"""
Извлечение мета-признаков таблицы для последующего выбора метода кластеризации:
разброс по колонкам, средние/медианы, число уникальных значений, кардинальность категорий,
сводка по масштабу и типам признаков.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_float(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def detect_format(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".csv":
        return "csv"
    if suf in {".tsv", ".txt"}:
        return "tsv"
    if suf in {".xlsx", ".xls"}:
        return "excel"
    if suf in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(f"Неподдерживаемое расширение: {suf}")


def read_table(path: Path, fmt: str | None, encoding: str | None, sep: str | None) -> pd.DataFrame:
    fmt = fmt or detect_format(path)
    if fmt == "csv":
        return pd.read_csv(path, encoding=encoding or "utf-8")
    if fmt == "tsv":
        return pd.read_csv(path, sep=sep or "\t", encoding=encoding or "utf-8")
    if fmt == "excel":
        return pd.read_excel(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Неизвестный формат: {fmt}")


def _shannon_entropy(counts: pd.Series) -> float:
    p = counts.astype(float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def numeric_clustering_stats(series: pd.Series, name: str) -> dict[str, Any]:
    s = series.astype(float)
    n = len(s)
    nulls = int(s.isna().sum())
    valid = s.dropna()
    n_valid = len(valid)
    n_unique = int(valid.nunique())

    base: dict[str, Any] = {
        "name": name,
        "kind": "numeric",
        "dtype": str(series.dtype),
        "n_rows": n,
        "null_count": nulls,
        "null_fraction": round(nulls / n, 6) if n else 0.0,
        "n_unique": n_unique,
        "unique_fraction": round(n_unique / n_valid, 6) if n_valid else 0.0,
    }

    if n_valid == 0:
        base["note"] = "все значения пропущены"
        return base

    vmin = float(valid.min())
    vmax = float(valid.max())
    mean = float(valid.mean())
    std = float(valid.std(ddof=1)) if n_valid > 1 else 0.0
    median = float(valid.median())
    q25, q75 = float(valid.quantile(0.25)), float(valid.quantile(0.75))

    base.update(
        {
            "min": vmin,
            "max": vmax,
            "range": vmax - vmin,
            "mean": mean,
            "median": median,
            "std": std,
            "variance": std**2,
            "iqr": q75 - q25,
            "skewness": float(valid.skew()) if n_valid > 2 else None,
            "kurtosis": float(valid.kurtosis()) if n_valid > 3 else None,
            "cv": _safe_float(std / abs(mean)) if abs(mean) > 1e-12 else None,
        }
    )
    return base


def categorical_clustering_stats(series: pd.Series, name: str) -> dict[str, Any]:
    n = len(series)
    nulls = int(series.isna().sum())
    non_null = series.dropna()
    n_valid = len(non_null)
    vc = non_null.value_counts()
    n_unique = int(vc.shape[0])

    entropy = _shannon_entropy(vc) if n_unique else 0.0
    max_entropy = math.log(n_unique) if n_unique > 1 else 0.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "name": name,
        "kind": "categorical",
        "dtype": str(series.dtype),
        "n_rows": n,
        "null_count": nulls,
        "null_fraction": round(nulls / n, 6) if n else 0.0,
        "n_unique": n_unique,
        "unique_fraction": round(n_unique / n_valid, 6) if n_valid else 0.0,
        "cardinality_ratio": round(n_unique / n, 6) if n else 0.0,
        "shannon_entropy": round(entropy, 6),
        "normalized_entropy": round(norm_entropy, 6),
    }


def datetime_clustering_stats(series: pd.Series, name: str) -> dict[str, Any]:
    dt = pd.to_datetime(series, errors="coerce")
    n = len(dt)
    nulls = int(dt.isna().sum())
    valid = dt.dropna()
    n_valid = len(valid)
    n_unique = int(valid.nunique())

    out: dict[str, Any] = {
        "name": name,
        "kind": "datetime",
        "dtype": str(series.dtype),
        "n_rows": n,
        "null_count": nulls,
        "null_fraction": round(nulls / n, 6) if n else 0.0,
        "n_unique": n_unique,
        "unique_fraction": round(n_unique / n_valid, 6) if n_valid else 0.0,
    }
    if n_valid == 0:
        out["note"] = "нет валидных дат"
        return out

    vmin, vmax = valid.min(), valid.max()
    span_ns = (vmax - vmin).total_seconds() * 1e9
    out["min"] = vmin.isoformat()
    out["max"] = vmax.isoformat()
    out["range_seconds"] = float(span_ns / 1e9)
    return out


def try_coerce_numeric(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(series):
        return series
    coerced = pd.to_numeric(series, errors="coerce")
    valid_ratio = float(coerced.notna().mean())
    if valid_ratio < 0.85:
        return None
    if coerced.notna().sum() == 0:
        return None
    return coerced


def column_clustering_profile(series: pd.Series) -> dict[str, Any]:
    name = str(series.name)

    if pd.api.types.is_datetime64_any_dtype(series):
        return datetime_clustering_stats(series, name)

    num = try_coerce_numeric(series)
    if num is not None:
        return numeric_clustering_stats(num, name)

    return categorical_clustering_stats(series.astype(str), name)


def build_clustering_summary(columns: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [c for c in columns if c.get("kind") == "numeric"]
    categorical = [c for c in columns if c.get("kind") == "categorical"]
    dt_cols = [c for c in columns if c.get("kind") == "datetime"]

    stds = [c["std"] for c in numeric if c.get("std") is not None and c["std"] > 1e-15]
    scale_ratio = None
    if len(stds) >= 2:
        mn, mx = min(stds), max(stds)
        scale_ratio = float(mx / mn) if mn > 0 else None

    avg_null = (
        sum(c.get("null_fraction", 0) for c in columns) / len(columns) if columns else 0.0
    )
    max_card = max((c.get("n_unique", 0) for c in categorical), default=0)

    if not numeric and categorical:
        mix = "categorical_only"
    elif numeric and not categorical:
        mix = "numeric_only"
    elif numeric and categorical:
        mix = "mixed"
    else:
        mix = "other"

    scale_level = "unknown"
    if scale_ratio is not None:
        scale_level = "high" if scale_ratio > 10 else "moderate" if scale_ratio > 3 else "low"

    missing_level = "low"
    if avg_null > 0.2:
        missing_level = "high"
    elif avg_null > 0.05:
        missing_level = "moderate"

    hints: list[str] = []
    if numeric and scale_level == "high":
        if mix == "numeric_only":
            hints.append(
                "Числовые признаки сильно различаются по масштабу (max std / min std): "
                "перед K-means / GMM нормируйте (StandardScaler) или взвесьте признаки."
            )
        elif mix == "mixed":
            hints.append(
                "Среди числовых колонок большой разброс масштабов: нормируйте числовую часть перед "
                "метриками расстояния вместе с закодированными категориями."
            )
    if mix == "mixed":
        hints.append(
            "Смешанные типы: чистый K-means по сырым данным обычно не подходит; "
            "кодирование категорий (one-hot, target), k-prototypes, Gower, или раздельные модели."
        )
    if categorical and max_card > 50:
        hints.append(
            "Высокая кардинальность категорий: при one-hot размерность вырастет; "
            "рассмотрите embedding, frequency/target encoding, feature hashing."
        )
    if missing_level in ("moderate", "high"):
        hints.append(
            "Заметная доля пропусков: явная импутация или алгоритмы с устойчивостью к NA "
            "(например, отдельная обработка до кластеризации)."
        )
    if any(
        c.get("skewness") is not None and abs(float(c["skewness"])) > 2 for c in numeric
    ):
        hints.append(
            "Сильная асимметрия в числовых колонках: для GMM предпочтительнее ближе к нормальному "
            "распределению (лог-преобразование) или робастные методы."
        )

    return {
        "n_numeric_features": len(numeric),
        "n_categorical_features": len(categorical),
        "n_datetime_features": len(dt_cols),
        "feature_mix": mix,
        "numeric_column_names": [c["name"] for c in numeric],
        "categorical_column_names": [c["name"] for c in categorical],
        "datetime_column_names": [c["name"] for c in dt_cols],
        "std_max_over_min": scale_ratio,
        "scale_heterogeneity": scale_level,
        "avg_null_fraction_across_columns": round(avg_null, 6),
        "missing_data_level": missing_level,
        "max_categorical_n_unique": max_card,
        "notes_for_clustering": hints,
    }


def extract_metadata(df: pd.DataFrame, source: Path, file_format: str) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    columns = [column_clustering_profile(df[c]) for c in df.columns]

    return {
        "meta": {
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(source.resolve()),
            "format": file_format,
            "n_rows": int(n_rows),
            "n_columns": int(n_cols),
        },
        "clustering_summary": build_clustering_summary(columns),
        "columns": columns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Мета-признаки таблицы для выбора кластеризации (разброс, средние, уникальные)."
    )
    parser.add_argument("path", type=Path, help="Путь к файлу (csv/tsv/xlsx/parquet)")
    parser.add_argument(
        "-f",
        "--format",
        choices=("csv", "tsv", "excel", "parquet"),
        default=None,
        help="Формат (по умолчанию — по расширению)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Записать JSON в файл вместо stdout",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Кодировка для CSV/TSV (по умолчанию utf-8)",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Разделитель для TSV (по умолчанию табуляция)",
    )
    args = parser.parse_args()

    if not args.path.is_file():
        print(f"Файл не найден: {args.path}", file=sys.stderr)
        return 1

    try:
        fmt = args.format or detect_format(args.path)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    try:
        df = read_table(args.path, fmt, args.encoding if fmt in ("csv", "tsv") else None, args.sep)
    except Exception as e:
        print(f"Ошибка чтения: {e}", file=sys.stderr)
        return 1

    payload = extract_metadata(df, args.path, fmt)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
