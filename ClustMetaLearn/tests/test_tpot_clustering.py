import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from clustmetalearn.tpot_clustering.encoding import SearchSpace, individual_to_pipeline
from clustmetalearn.tpot_clustering.evolve import EvolutionConfig, run_evolution


def test_individual_to_pipeline_runs():
    space = SearchSpace.from_shape(n_samples=80, n_features=4)
    ind = [1, 0, 2, 0, 3, 0]
    pipe = individual_to_pipeline(ind, space, random_state=0)
    X, _ = make_blobs(n_samples=60, centers=3, n_features=4, random_state=1)
    pipe.fit(X)
    labels = pipe.predict(X)
    assert len(labels) == 60
    assert len(np.unique(labels)) >= 1


def test_run_evolution_small():
    X, _y = make_blobs(n_samples=90, centers=3, n_features=2, random_state=2)
    space = SearchSpace.from_shape(X.shape[0], X.shape[1])
    cfg = EvolutionConfig(
        generations=1,
        population_size=8,
        cv_splits=3,
        random_state=3,
        metric="silhouette",
    )
    result = run_evolution(X, None, space, cfg)
    assert result.best_fitness > -1e8
    result.best_pipeline.fit(X)
    assert hasattr(result.best_pipeline, "predict")


def test_cli_skips_on_small_data(tmp_path: Path):
    X, y = make_blobs(n_samples=3, centers=2, n_features=2, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b"])
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False)
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "clustmetalearn.tpot_clustering",
        str(p),
        "--generations",
        "1",
        "--population",
        "4",
        "--cv",
        "3",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    assert r.returncode != 0
    out = (r.stderr + r.stdout).lower()
    assert "more rows" in out


def test_cli_runs_on_synthetic_csv(tmp_path: Path):
    X, y = make_blobs(n_samples=120, centers=3, n_features=2, random_state=5)
    df = pd.DataFrame(X, columns=["a", "b"])
    df["label"] = y
    p = tmp_path / "blob.csv"
    df.to_csv(p, index=False)
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "clustmetalearn.tpot_clustering",
        str(p),
        "-l",
        "label",
        "--metric",
        "ari",
        "--generations",
        "1",
        "--population",
        "6",
        "--cv",
        "3",
        "--seed",
        "7",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "Best CV score" in r.stdout


def test_cli_runs_on_synthetic_bin(tmp_path: Path):
    X, y = make_blobs(n_samples=90, centers=3, n_features=2, random_state=6)
    data_path = tmp_path / "data.bin"
    label_path = tmp_path / "label.bin"
    X.astype(np.float32).tofile(data_path)
    y.astype(np.int32).tofile(label_path)
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "clustmetalearn.tpot_clustering",
        str(data_path),
        "--input-format",
        "bin",
        "--n-samples",
        str(X.shape[0]),
        "--n-features",
        str(X.shape[1]),
        "--label-bin",
        str(label_path),
        "--metric",
        "ari",
        "--generations",
        "1",
        "--population",
        "6",
        "--cv",
        "3",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "Best CV score" in r.stdout
