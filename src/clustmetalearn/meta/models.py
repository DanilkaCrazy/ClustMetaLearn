"""Persisted meta-models: CVIsel and AlgRank."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from clustmetalearn.meta.constants import META_FEATURE_COLS


@dataclass
class MetaModelBundle:
    cvisel: RandomForestClassifier | None
    algrank: RandomForestClassifier | None
    feature_cols: list[str]
    cvi_classes: list[str]
    algo_classes: list[str]
    use_topo: bool


def _select_feature_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:5]}")
    return df[list(feature_cols)].to_numpy(dtype=float)


def train_cvisel(
    df: pd.DataFrame,
    *,
    target_col: str = "target_cvi",
    feature_cols: Sequence[str] | None = None,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, list[str], list[str]]:
    feature_cols = list(feature_cols or META_FEATURE_COLS)
    train_mask = df["split"] == "train" if "split" in df.columns else pd.Series(True, index=df.index)
    sub = df.loc[train_mask]
    X = _select_feature_matrix(sub, feature_cols)
    y = sub[target_col].astype(str).to_numpy()
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    clf.fit(X, y)
    classes = list(clf.classes_)
    return clf, feature_cols, classes


def train_algrank(
    df: pd.DataFrame,
    *,
    target_col: str = "best_algorithm",
    feature_cols: Sequence[str] | None = None,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, list[str], list[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Column {target_col!r} required for AlgRank training.")
    feature_cols = list(feature_cols or META_FEATURE_COLS)
    train_mask = df["split"] == "train" if "split" in df.columns else pd.Series(True, index=df.index)
    sub = df.loc[train_mask].dropna(subset=[target_col])
    if sub.empty:
        raise ValueError("No training rows with algorithm labels.")
    X = _select_feature_matrix(sub, feature_cols)
    y = sub[target_col].astype(str).to_numpy()
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    clf.fit(X, y)
    return clf, feature_cols, list(clf.classes_)


def predict_top_k(
    clf: RandomForestClassifier,
    X: np.ndarray,
    *,
    k: int = 3,
) -> list[list[str]]:
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    out: list[list[str]] = []
    for row in proba:
        order = np.argsort(row)[::-1][:k]
        out.append([classes[i] for i in order])
    return out


def save_bundle(bundle: MetaModelBundle, models_dir: Path) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "feature_cols": bundle.feature_cols,
        "cvi_classes": bundle.cvi_classes,
        "algo_classes": bundle.algo_classes,
        "use_topo": bundle.use_topo,
    }
    (models_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if bundle.cvisel is not None:
        joblib.dump(bundle.cvisel, models_dir / "cvisel.joblib")
    if bundle.algrank is not None:
        joblib.dump(bundle.algrank, models_dir / "algrank.joblib")


def load_bundle(models_dir: Path) -> MetaModelBundle:
    models_dir = Path(models_dir)
    meta = json.loads((models_dir / "meta.json").read_text(encoding="utf-8"))
    cvisel_path = models_dir / "cvisel.joblib"
    algrank_path = models_dir / "algrank.joblib"
    return MetaModelBundle(
        cvisel=joblib.load(cvisel_path) if cvisel_path.is_file() else None,
        algrank=joblib.load(algrank_path) if algrank_path.is_file() else None,
        feature_cols=meta["feature_cols"],
        cvi_classes=meta.get("cvi_classes", []),
        algo_classes=meta.get("algo_classes", []),
        use_topo=bool(meta.get("use_topo", True)),
    )


def feature_vector_from_dict(feats: dict[str, float], feature_cols: Sequence[str]) -> np.ndarray:
    return np.array([[float(feats[c]) for c in feature_cols]], dtype=float)
