"""Train CVIsel / AlgRank and write model artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clustmetalearn.meta.clm import assign_clm_splits
from clustmetalearn.meta.constants import META_FEATURE_COLS, NON_TOPO_FEATURE_COLS, TOPO_FEATURE_COLS
from clustmetalearn.meta.hp_intervals import compute_hp_intervals, save_hp_intervals
from clustmetalearn.meta.io import load_meta_table
from clustmetalearn.meta.models import MetaModelBundle, save_bundle, train_algrank, train_cvisel


def prepare_training_table(
    meta_csv: Path,
    *,
    labels_csv: Path | None = None,
    use_topo: bool = True,
) -> pd.DataFrame:
    df = load_meta_table(meta_csv)
    df = assign_clm_splits(df)
    if labels_csv is not None and Path(labels_csv).is_file():
        labels = pd.read_csv(labels_csv)
        df = df.merge(labels, on="dataset", how="left")
    return df


def train_meta_models(
    meta_csv: Path,
    models_dir: Path,
    *,
    labels_csv: Path | None = None,
    use_topo: bool = True,
    random_state: int = 42,
) -> MetaModelBundle:
    df = prepare_training_table(meta_csv, labels_csv=labels_csv, use_topo=use_topo)
    feature_cols = list(META_FEATURE_COLS if use_topo else NON_TOPO_FEATURE_COLS)

    cvisel = None
    cvi_classes: list[str] = []
    if "target_cvi" in df.columns and df["target_cvi"].notna().any():
        cvisel, feature_cols, cvi_classes = train_cvisel(
            df,
            target_col="target_cvi",
            feature_cols=feature_cols,
            random_state=random_state,
        )

    algrank = None
    algo_classes: list[str] = []
    if "best_algorithm" in df.columns and df["best_algorithm"].notna().sum() >= 3:
        algrank, feature_cols, algo_classes = train_algrank(
            df,
            target_col="best_algorithm",
            feature_cols=feature_cols,
            random_state=random_state,
        )
        intervals = compute_hp_intervals(df.dropna(subset=["best_algorithm"]))
        save_hp_intervals(intervals, Path(models_dir) / "hp_intervals.json")

    bundle = MetaModelBundle(
        cvisel=cvisel,
        algrank=algrank,
        feature_cols=feature_cols,
        cvi_classes=cvi_classes,
        algo_classes=algo_classes,
        use_topo=use_topo,
    )
    save_bundle(bundle, models_dir)
    return bundle
