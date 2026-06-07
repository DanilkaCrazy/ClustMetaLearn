"""Evaluation protocol: accuracy, top-k, low-CLM slice, topo ablation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from clustmetalearn.meta.constants import NON_TOPO_FEATURE_COLS, TOPO_FEATURE_COLS
from clustmetalearn.meta.models import (
    _select_feature_matrix,
    predict_top_k,
    train_algrank,
    train_cvisel,
)


@dataclass
class EvalReport:
    task: str
    split: str
    n_samples: int
    accuracy: float
    f1_weighted: float
    top3_accuracy: float | None = None
    details: dict = field(default_factory=dict)


def _top_k_accuracy(clf: RandomForestClassifier, X: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    tops = predict_top_k(clf, X, k=k)
    hits = sum(1 for pred_list, yt in zip(tops, y_true, strict=True) if yt in pred_list)
    return hits / len(y_true) if len(y_true) else 0.0


def evaluate_classifier(
    clf: RandomForestClassifier,
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    split_name: str,
    task: str,
    top_k: int | None = 3,
) -> EvalReport:
    mask = df["split"] == split_name
    sub = df.loc[mask].dropna(subset=[target_col])
    if sub.empty:
        return EvalReport(task=task, split=split_name, n_samples=0, accuracy=0.0, f1_weighted=0.0)
    X = _select_feature_matrix(sub, feature_cols)
    y = sub[target_col].astype(str).to_numpy()
    y_pred = clf.predict(X)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, average="weighted", zero_division=0))
    top3 = _top_k_accuracy(clf, X, y, k=top_k) if top_k else None
    return EvalReport(
        task=task,
        split=split_name,
        n_samples=len(sub),
        accuracy=acc,
        f1_weighted=f1,
        top3_accuracy=top3,
        details={"classification_report": classification_report(y, y_pred, zero_division=0)},
    )


def evaluate_low_clm(
    clf: RandomForestClassifier,
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    task: str,
) -> EvalReport:
    if "CLM" not in df.columns:
        return EvalReport(task=task, split="low_clm", n_samples=0, accuracy=0.0, f1_weighted=0.0)
    q1 = df["CLM"].quantile(1 / 3)
    sub = df[(df["CLM"] < q1) & df[target_col].notna()]
    if sub.empty:
        return EvalReport(task=task, split="low_clm", n_samples=0, accuracy=0.0, f1_weighted=0.0)
    X = _select_feature_matrix(sub, feature_cols)
    y = sub[target_col].astype(str).to_numpy()
    y_pred = clf.predict(X)
    return EvalReport(
        task=task,
        split="low_clm",
        n_samples=len(sub),
        accuracy=float(accuracy_score(y, y_pred)),
        f1_weighted=float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        top3_accuracy=_top_k_accuracy(clf, X, y, k=3),
    )


def cross_val_train_score(
    clf: RandomForestClassifier,
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    cv: int = 3,
) -> float:
    sub = df[df["split"] == "train"].dropna(subset=[target_col])
    if len(sub) < cv + 1:
        return 0.0
    X = _select_feature_matrix(sub, feature_cols)
    y = sub[target_col].astype(str).to_numpy()
    scores = cross_val_score(
        clf,
        X,
        y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    return float(scores.mean())


def baseline_majority(df: pd.DataFrame, target_col: str, split: str) -> float:
    train = df[df["split"] == "train"][target_col].astype(str)
    test = df[df["split"] == split][target_col].astype(str)
    if train.empty or test.empty:
        return 0.0
    mode = train.mode().iloc[0]
    return float((test == mode).mean())


def topo_ablation_cvisel(
    df: pd.DataFrame,
    *,
    target_col: str = "target_cvi",
    random_state: int = 42,
) -> dict[str, float]:
    """Compare CVIsel train accuracy with and without topological features."""
    df = df.copy()
    if "split" not in df.columns:
        from clustmetalearn.meta.clm import assign_clm_splits

        df = assign_clm_splits(df)

    results: dict[str, float] = {}
    for name, cols in (
        ("with_topo", list(NON_TOPO_FEATURE_COLS) + list(TOPO_FEATURE_COLS)),
        ("without_topo", list(NON_TOPO_FEATURE_COLS)),
    ):
        clf, _, _ = train_cvisel(df, target_col=target_col, feature_cols=cols, random_state=random_state)
        sub = df[df["split"] == "train"].dropna(subset=[target_col])
        if sub.empty:
            results[name] = 0.0
            continue
        X = _select_feature_matrix(sub, cols)
        y = sub[target_col].astype(str).to_numpy()
        results[name] = float(accuracy_score(y, clf.predict(X)))
    results["delta_topo"] = results.get("with_topo", 0.0) - results.get("without_topo", 0.0)
    return results


def topo_ablation_algrank(
    df: pd.DataFrame,
    *,
    target_col: str = "best_algorithm",
    random_state: int = 42,
) -> dict[str, float]:
    if target_col not in df.columns:
        return {}
    df = df.copy()
    if "split" not in df.columns:
        from clustmetalearn.meta.clm import assign_clm_splits

        df = assign_clm_splits(df)
    results: dict[str, float] = {}
    for name, cols in (
        ("with_topo", list(NON_TOPO_FEATURE_COLS) + list(TOPO_FEATURE_COLS)),
        ("without_topo", list(NON_TOPO_FEATURE_COLS)),
    ):
        try:
            clf, _, _ = train_algrank(df, target_col=target_col, feature_cols=cols, random_state=random_state)
        except ValueError:
            return {}
        sub = df[df["split"] == "train"].dropna(subset=[target_col])
        if sub.empty:
            results[name] = 0.0
            continue
        X = _select_feature_matrix(sub, cols)
        y = sub[target_col].astype(str).to_numpy()
        results[name] = float(accuracy_score(y, clf.predict(X)))
    results["delta_topo"] = results.get("with_topo", 0.0) - results.get("without_topo", 0.0)
    return results


def format_report(reports: list[EvalReport], ablation: dict | None = None) -> str:
    lines: list[str] = []
    for r in reports:
        line = (
            f"[{r.task}] split={r.split} n={r.n_samples} "
            f"acc={r.accuracy:.3f} f1={r.f1_weighted:.3f}"
        )
        if r.top3_accuracy is not None:
            line += f" top3={r.top3_accuracy:.3f}"
        lines.append(line)
    if ablation:
        lines.append("--- topo ablation (train fit accuracy) ---")
        for k, v in ablation.items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return "\n".join(lines)
