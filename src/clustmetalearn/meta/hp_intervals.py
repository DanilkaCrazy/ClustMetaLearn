"""Empirical hyperparameter intervals from algorithm label tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def compute_hp_intervals(labels_df: pd.DataFrame) -> dict[str, dict]:
    """Per-algorithm empirical k range and scaler usage rate."""
    if "best_algorithm" not in labels_df.columns:
        return {}
    out: dict[str, dict] = {}
    for algo, grp in labels_df.groupby("best_algorithm"):
        k = grp["best_k"].astype(int)
        out[str(algo)] = {
            "k_min": int(k.min()),
            "k_max": int(k.max()),
            "k_median": float(k.median()),
            "scaler_rate": float(grp["use_scaler"].mean()) if "use_scaler" in grp.columns else None,
            "n_datasets": int(len(grp)),
        }
    return out


def save_hp_intervals(intervals: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(intervals, indent=2), encoding="utf-8")


def load_hp_intervals(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
