"""Pairwise learning-to-rank for candidate preselection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import SGDClassifier


@dataclass
class RankingTrickConfig:
    enabled: bool = False
    warmup_evals: int = 30
    oversample: int = 3
    max_pairs: int = 2000
    random_state: int = 0


class PairwiseLinearRanker:
    """Linear ranker on pairwise genome differences."""

    def __init__(self, *, random_state: int = 0) -> None:
        self.random_state = random_state
        self._clf: SGDClassifier | None = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit_from_archive(
        self,
        X: np.ndarray,
        y_fitness: np.ndarray,
        *,
        max_pairs: int,
    ) -> None:
        """Fit on random pairs (x_i - x_j, sign(f_i - f_j))."""
        n = X.shape[0]
        if n < 4:
            self._is_fitted = False
            return

        rng = np.random.default_rng(self.random_state)
        pairs = min(max_pairs, n * (n - 1) // 2)
        i = rng.integers(0, n, size=pairs)
        j = rng.integers(0, n, size=pairs)
        mask = i != j
        i, j = i[mask], j[mask]
        if i.size == 0:
            self._is_fitted = False
            return

        diff = X[i] - X[j]
        labels = (y_fitness[i] > y_fitness[j]).astype(int)

        if np.all(labels == labels[0]):
            self._is_fitted = False
            return

        clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=1000,
            tol=1e-3,
            random_state=self.random_state,
        )
        clf.fit(diff, labels)
        self._clf = clf
        self._is_fitted = True

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return ranking scores (higher = better)."""
        if not self._is_fitted or self._clf is None:
            return np.zeros(X.shape[0], dtype=float)
        return self._clf.decision_function(X)

