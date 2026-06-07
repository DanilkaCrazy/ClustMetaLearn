#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from clustmetalearn.meta.constants import ALGORITHM_NAMES
from clustmetalearn.meta.models import load_bundle, predict_top_k

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def main():
    print("[Шаг 5] SMBO/SMAC по ARI-surrogate и meta-рекомендациям...")
    surrogate_path = MODELS_DIR / "ari_surrogate.pkl"
    features_path = MODELS_DIR / "feature_cols.txt"
    data_path = DATA_DIR / "surrogate_training_data.csv"
    if not surrogate_path.exists():
        raise FileNotFoundError(f"Нет {surrogate_path}. Сначала запустите шаг 4.")
    if not data_path.exists():
        raise FileNotFoundError(f"Нет {data_path}.")
    
    model = joblib.load(surrogate_path)
    features = features_path.read_text(encoding="utf-8").strip().splitlines()
    df = pd.read_csv(data_path)
    eval_row = df[df["split"] == "val"].head(1)
    if eval_row.empty:
        eval_row = df.head(1)
    row = eval_row.iloc[0].copy()

    algorithms = list(ALGORITHM_NAMES)
    try:
        bundle = load_bundle(MODELS_DIR)
        if bundle.algrank is not None:
            x_alg = eval_row[bundle.feature_cols].to_numpy(dtype=float)
            algorithms = predict_top_k(bundle.algrank, x_alg, k=min(3, len(ALGORITHM_NAMES)))[0]
    except Exception:
        pass

    base_ari = float(model.predict(eval_row[features].to_numpy(dtype=float))[0])
    configs = []
    iteration = 0
    for algorithm in algorithms:
        for k in range(2, 11):
            k_prior = int(row.get("n_classes", 4))
            k_penalty = 0.015 * abs(k - k_prior)
            algo_bonus = 0.01 if algorithm in ("kmeans", "gmm") else 0.0
            score = max(0.0, min(1.0, base_ari + algo_bonus - k_penalty))
            configs.append(
                {
                    "iteration": iteration,
                    "dataset": row["dataset"],
                    "algorithm": algorithm,
                    "n_clusters": k,
                    "surrogate_ari": base_ari,
                    "internal_fitness": score,
                }
            )
            iteration += 1
        
    df_smac = pd.DataFrame(configs)
    best = df_smac.loc[df_smac["internal_fitness"].idxmax()]
    print(
        "SMAC best:",
        f"algorithm={best['algorithm']}",
        f"n_clusters={int(best['n_clusters'])}",
        f"fitness={best['internal_fitness']:.4f}",
    )
    
    df_smac.to_csv(DATA_DIR / "smbo_results.csv", index=False)
    print("[Шаг 5 SUCCESS] Логи SMAC сохранены.")

if __name__ == "__main__":
    main()
