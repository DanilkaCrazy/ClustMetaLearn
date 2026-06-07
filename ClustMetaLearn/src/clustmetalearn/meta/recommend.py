"""Inference: meta-features to CVI and algorithm recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from clustmetalearn.meta.constants import CVI_CLASS_TO_METRIC
from clustmetalearn.meta.features import (
    extract_meta_features,
    extract_meta_features_from_bin,
    extract_meta_features_from_csv,
)
from clustmetalearn.meta.hp_intervals import load_hp_intervals
from clustmetalearn.meta.models import (
    MetaModelBundle,
    feature_vector_from_dict,
    load_bundle,
    predict_top_k,
)


@dataclass(frozen=True)
class MetaRecommendation:
    meta_features: dict[str, float]
    predicted_cvi: str | None
    predicted_cvi_metric: str | None
    predicted_algorithm: str | None
    top3_algorithms: tuple[str, ...]
    hp_intervals: dict | None


def recommend_from_features(
    feats: dict[str, float],
    bundle: MetaModelBundle,
    *,
    hp_intervals: dict | None = None,
) -> MetaRecommendation:
    X = feature_vector_from_dict(feats, bundle.feature_cols)
    cvi_name = None
    metric = None
    if bundle.cvisel is not None:
        cvi_name = str(bundle.cvisel.predict(X)[0])
        metric = CVI_CLASS_TO_METRIC.get(cvi_name)

    algo = None
    top3: tuple[str, ...] = ()
    if bundle.algrank is not None:
        algo = str(bundle.algrank.predict(X)[0])
        top3 = tuple(predict_top_k(bundle.algrank, X, k=3)[0])

    hp = None
    if hp_intervals and algo and algo in hp_intervals:
        hp = hp_intervals[algo]

    return MetaRecommendation(
        meta_features=feats,
        predicted_cvi=cvi_name,
        predicted_cvi_metric=metric,
        predicted_algorithm=algo,
        top3_algorithms=top3,
        hp_intervals=hp,
    )


def recommend_from_table(
    csv_path: str,
    models_dir: Path,
    *,
    label_column: str | None = None,
    include_topology: bool = True,
) -> MetaRecommendation:
    bundle = load_bundle(models_dir)
    if include_topology != bundle.use_topo:
        include_topology = bundle.use_topo
    feats = extract_meta_features_from_csv(
        csv_path,
        label_column=label_column,
        include_topology=include_topology,
    )
    hp_path = Path(models_dir) / "hp_intervals.json"
    hp = load_hp_intervals(hp_path) if hp_path.is_file() else None
    return recommend_from_features(feats, bundle, hp_intervals=hp)


def recommend_from_bin(
    data_path: str,
    models_dir: Path,
    *,
    n_samples: int,
    n_features: int,
    dtype: str = "float32",
    include_topology: bool = True,
) -> MetaRecommendation:
    bundle = load_bundle(models_dir)
    if include_topology != bundle.use_topo:
        include_topology = bundle.use_topo
    feats = extract_meta_features_from_bin(
        data_path,
        n_samples=n_samples,
        n_features=n_features,
        dtype=dtype,
        include_topology=include_topology,
    )
    hp_path = Path(models_dir) / "hp_intervals.json"
    hp = load_hp_intervals(hp_path) if hp_path.is_file() else None
    return recommend_from_features(feats, bundle, hp_intervals=hp)


def recommend_from_array(
    X: np.ndarray,
    models_dir: Path,
    *,
    include_topology: bool = True,
) -> MetaRecommendation:
    bundle = load_bundle(models_dir)
    feats = extract_meta_features(X, include_topology=include_topology)
    hp_path = Path(models_dir) / "hp_intervals.json"
    hp = load_hp_intervals(hp_path) if hp_path.is_file() else None
    return recommend_from_features(feats, bundle, hp_intervals=hp)
