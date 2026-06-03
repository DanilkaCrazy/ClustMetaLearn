"""Dataset I/O: labeled .bin repositories and meta-feature tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def list_labeled_datasets(root: Path) -> list[str]:
    root = Path(root)
    names: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "data.bin").is_file() and (entry / "label.bin").is_file():
            names.append(entry.name)
    return names


def load_labeled_dataset(
    dataset_dir: Path,
    *,
    summary_row: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load X, y from data.bin / label.bin (hj-n/labeled-datasets layout)."""
    dataset_dir = Path(dataset_dir)
    data_path = dataset_dir / "data.bin"
    label_path = dataset_dir / "label.bin"
    if not data_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing data.bin or label.bin in {dataset_dir}")

    data = np.fromfile(data_path, dtype=np.float32)
    labels = np.fromfile(label_path, dtype=np.int32)
    n = len(labels)
    if n == 0:
        raise ValueError(f"Empty labels in {dataset_dir}")

    if summary_row is not None:
        n_samples = int(summary_row["n_samples"])
        n_features = int(summary_row["n_features"])
        expected = n_samples * n_features
        if expected != len(data):
            raise ValueError(
                f"Shape mismatch for {dataset_dir.name}: "
                f"summary {n_samples}x{n_features} vs {len(data)} floats"
            )
        X = data.reshape(n_samples, n_features)
        if len(labels) != n_samples:
            raise ValueError(f"Label count {len(labels)} != n_samples {n_samples}")
        return X.astype(np.float64), labels

    if len(data) % n != 0:
        raise ValueError(
            f"Cannot infer feature shape for {dataset_dir.name}: "
            f"{len(data)} floats, {n} labels. Pass dataset_summary.csv."
        )
    n_features = len(data) // n
    X = data.reshape(n, n_features)
    return X.astype(np.float64), labels


def load_dataset_summary(summary_path: Path) -> pd.DataFrame:
    return pd.read_csv(summary_path)


def load_meta_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        raise ValueError("Meta table must contain a 'dataset' column.")
    return df


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
