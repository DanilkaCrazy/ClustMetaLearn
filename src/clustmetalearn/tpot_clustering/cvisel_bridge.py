"""Use trained CVIsel model to pick evolution fitness metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clustmetalearn.meta.constants import CVI_CLASS_TO_METRIC
from clustmetalearn.meta.features import extract_meta_features
from clustmetalearn.meta.models import feature_vector_from_dict, load_bundle
from clustmetalearn.tpot_clustering.fitness import MetricName


def metric_from_cvisel(
    X: np.ndarray,
    models_dir: Path,
    *,
    fallback: MetricName = "silhouette",
) -> MetricName:
    bundle = load_bundle(models_dir)
    if bundle.cvisel is None:
        return fallback
    feats = extract_meta_features(X, include_topology=bundle.use_topo)
    vec = feature_vector_from_dict(feats, bundle.feature_cols)
    cvi_class = str(bundle.cvisel.predict(vec)[0])
    return CVI_CLASS_TO_METRIC.get(cvi_class, fallback)  # type: ignore[return-value]
