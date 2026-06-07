#!/usr/bin/env python3
from pathlib import Path

from clustmetalearn.meta.schema import write_normalized_tables

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def main():
    print("[Шаг 2] Нормализация мета-признаков и целевых колонок...")
    paths = write_normalized_tables(BASE_DIR)
    for name, path in paths.items():
        print(f"[Шаг 2 SUCCESS] {name}: {path}")

if __name__ == "__main__":
    main()
