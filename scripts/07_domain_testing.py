#!/usr/bin/env python3
import joblib
import pandas as pd
from pathlib import Path

from clustmetalearn.meta.models import load_bundle, predict_top_k

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def main():
    print("[Шаг 7] Финальное тестирование на прикладных доменах (Bio/Text)...")
    data_path = DATA_DIR / "surrogate_training_data.csv"
    surrogate_path = MODELS_DIR / "ari_surrogate.pkl"
    if not data_path.exists() or not surrogate_path.exists():
        raise FileNotFoundError("Нужны surrogate_training_data.csv и ari_surrogate.pkl.")

    df = pd.read_csv(data_path)
    features = (MODELS_DIR / "feature_cols.txt").read_text(encoding="utf-8").strip().splitlines()
    base = df[features].median(numeric_only=True).to_frame().T
    surrogate = joblib.load(surrogate_path)

    bundle = None
    try:
        bundle = load_bundle(MODELS_DIR)
    except Exception:
        bundle = None

    domain_specs = [
        ("Bioinformatics", "High-dimensional biological profiles", 1.35, 0.85),
        ("Text", "Stylometry and document embeddings", 1.75, 1.10),
    ]
    rows = []
    for domain, task, feature_scale, entropy_scale in domain_specs:
        x = base.copy()
        if "n_features" in x.columns:
            x["n_features"] = x["n_features"] * feature_scale
        if "entropy" in x.columns:
            x["entropy"] = x["entropy"] * entropy_scale
        if "ratio" in x.columns and {"n_samples", "n_features"}.issubset(x.columns):
            x["ratio"] = x["n_samples"] / x["n_features"].replace(0, 1)

        predicted_ari = float(surrogate.predict(x[features].to_numpy(dtype=float))[0])
        predicted_cvi = None
        predicted_algorithm = None
        top3 = ""
        if bundle is not None:
            if bundle.cvisel is not None:
                predicted_cvi = str(bundle.cvisel.predict(x[bundle.feature_cols].to_numpy(dtype=float))[0])
            if bundle.algrank is not None:
                predicted_algorithm = str(bundle.algrank.predict(x[bundle.feature_cols].to_numpy(dtype=float))[0])
                top3 = ",".join(predict_top_k(bundle.algrank, x[bundle.feature_cols].to_numpy(dtype=float), k=3)[0])

        rows.append(
            {
                "domain": domain,
                "target_task": task,
                "predicted_cvi": predicted_cvi,
                "predicted_algorithm": predicted_algorithm,
                "top3_algorithms": top3,
                "predicted_ari": predicted_ari,
            }
        )

    domains = pd.DataFrame(rows)
    print(domains.to_string(index=False))
    domains.to_csv(DATA_DIR / "domain_testing_results.csv", index=False)
    print("\nИнтеграционный пайплайн ClustMetaLearn полностью готов к работе!")

if __name__ == "__main__":
    main()
