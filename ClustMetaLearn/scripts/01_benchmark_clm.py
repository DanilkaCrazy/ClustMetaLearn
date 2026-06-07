#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

from clustmetalearn.meta.schema import (
    dataset_summary_from_meta,
    default_surrogate_source,
    normalize_meta_dataframe,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def main():
    print("[Шаг 1] Подготовка сводки датасетов и CLM-разбиения...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    source = default_surrogate_source(BASE_DIR)
    df = normalize_meta_dataframe(pd.read_csv(source))
    summary = dataset_summary_from_meta(df)

    output_csv = DATA_DIR / "dataset_summary.csv"
    summary.to_csv(output_csv, index=False)
    print(f"[Шаг 1 SUCCESS] Source: {source}")
    print(f"[Шаг 1 SUCCESS] Rows: {len(summary)}")
    print(f"[Шаг 1 SUCCESS] Сводная таблица сохранена в: {output_csv}")

if __name__ == "__main__":
    main()