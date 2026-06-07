#!/usr/bin/env python3
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def main():
    print("[Шаг 6] Интерпретация важности мета-признаков...")
    rows = []
    candidates = [
        ("cvisel", MODELS_DIR / "cvisel.joblib"),
        ("algrank", MODELS_DIR / "algrank.joblib"),
        ("ari_surrogate", MODELS_DIR / "ari_surrogate.pkl"),
    ]
    features = []
    feature_file = MODELS_DIR / "feature_cols.txt"
    if feature_file.exists():
        features = feature_file.read_text(encoding="utf-8").strip().splitlines()

    meta_json = MODELS_DIR / "meta.json"
    if meta_json.exists() and not features:
        import json

        features = json.loads(meta_json.read_text(encoding="utf-8")).get("feature_cols", [])

    for model_name, path in candidates:
        if not path.exists():
            continue
        model = joblib.load(path)
        if not hasattr(model, "feature_importances_"):
            continue
        model_features = features
        if model_name in ("cvisel", "algrank") and meta_json.exists():
            import json

            model_features = json.loads(meta_json.read_text(encoding="utf-8")).get("feature_cols", features)
        for feature, importance in zip(model_features, model.feature_importances_, strict=False):
            rows.append(
                {
                    "model": model_name,
                    "meta_feature": feature,
                    "importance": float(importance),
                }
            )

    if not rows:
        raise FileNotFoundError("Нет моделей с feature_importances_.")

    df_importance = pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])
    print(df_importance.to_string(index=False))
    df_importance.to_csv(DATA_DIR / "feature_importance_cvisel.csv", index=False)
    print("[Шаг 6 SUCCESS] Важность признаков сохранена.")

if __name__ == "__main__":
    main()
