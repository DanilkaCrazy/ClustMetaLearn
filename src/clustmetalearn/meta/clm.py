"""CLM-based dataset splits (train / val / test tertiles)."""

from __future__ import annotations

import pandas as pd


def assign_clm_splits(
    meta_df: pd.DataFrame,
    *,
    clm_column: str = "CLM",
    dataset_column: str = "dataset",
) -> pd.DataFrame:
    """Add split column: train (high CLM), val (mid), test (low)."""
    if clm_column not in meta_df.columns:
        raise ValueError(f"Column {clm_column!r} not found.")
    df = meta_df.copy()
    n = len(df)
    if n == 0:
        df["split"] = []
        return df
    rank = df[clm_column].rank(method="first")
    df["split"] = "val"
    df.loc[rank <= n / 3, "split"] = "test"
    df.loc[rank > (2 * n) / 3, "split"] = "train"
    return df


def split_masks(meta_df: pd.DataFrame) -> dict[str, pd.Series]:
    if "split" not in meta_df.columns:
        meta_df = assign_clm_splits(meta_df)
    return {
        name: meta_df["split"] == name
        for name in ("train", "val", "test")
    }
