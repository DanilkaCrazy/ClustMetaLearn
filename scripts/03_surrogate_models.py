#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def main():
    print("[Шаг 3] Обучение суррогатных моделей производительности CVIsel...")
    input_path = DATA_DIR / "surrogate_training_data.csv"
    
    if not input_path.exists():
        print("❌ Ошибка: нет данных для обучения.")
        return
        
    df = pd.read_csv(input_path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    features = ["n_samples", "n_features", "n_classes", "CLM", "skewness", "kurtosis", "shannon_entropy", "betti_0", "betti_1"]
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(train_df[features].values, train_df["ari"].values)
    
    val_score = model.score(val_df[features].values, val_df["ari"].values)
    print(f"Качество суррогатной модели R^2 на валидации: {val_score:.3f}")
    
    joblib.dump(model, MODELS_DIR / "cvisel_rf.pkl")
    with open(MODELS_DIR / "feature_cols.txt", "w") as f:
        f.write("\n".join(features))
    print("[Шаг 3 SUCCESS] Веса суррогатной модели сохранены.")

if __name__ == "__main__":
    main()