#!/usr/bin/env python3
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from clustmetalearn.meta.constants import META_FEATURE_COLS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

def main():
    print("[Шаг 4] Обучение ARI-surrogate...")
    input_path = DATA_DIR / "surrogate_training_data.csv"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Нет {input_path}. Сначала запустите шаг 2.")
        
    df = pd.read_csv(input_path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    features = [c for c in META_FEATURE_COLS if c in df.columns]
    if "ari" not in df.columns:
        raise ValueError("surrogate_training_data.csv must contain ari column.")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    if train_df.empty:
        train_df = df.copy()
    if val_df.empty:
        val_df = train_df.copy()
    
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(train_df[features].values, train_df["ari"].values)
    
    pred = model.predict(val_df[features].values)
    r2 = r2_score(val_df["ari"].values, pred) if len(val_df) > 1 else 0.0
    mae = mean_absolute_error(val_df["ari"].values, pred)
    print(f"ARI-surrogate validation R^2: {r2:.3f}")
    print(f"ARI-surrogate validation MAE: {mae:.3f}")
    
    joblib.dump(model, MODELS_DIR / "ari_surrogate.pkl")
    with open(MODELS_DIR / "feature_cols.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(features))
    pd.DataFrame(
        [{"model": "ari_surrogate", "r2": r2, "mae": mae, "n_train": len(train_df), "n_val": len(val_df)}]
    ).to_csv(DATA_DIR / "surrogate_model_metrics.csv", index=False)
    print("[Шаг 4 SUCCESS] Модель сохранена:", MODELS_DIR / "ari_surrogate.pkl")

if __name__ == "__main__":
    main()