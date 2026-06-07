#!/usr/bin/env python3
import sys
import os
import subprocess
import time
from pathlib import Path

STEPS = [
    ("01_benchmark_clm.py", "Сбор бенчмарков и CLM-фильтрация"),
    ("02_isa_adjusted_ivms.py", "Нормализация мета-признаков и ISA-таргетов"),
    ("03_meta_models.py", "Обучение CVIsel / AlgRank"),
    ("03_surrogate_models.py", "Обучение ARI-surrogate"),
    ("04_smac_tuning.py", "Настройка SMBO/SMAC для гиперпараметров"),
    ("05_time_budget_validation.py", "Валидация стратегий Time Budget"),
    ("06_shap_analysis.py", "Интерпретация важности признаков"),
    ("07_domain_testing.py", "Тестирование на прикладных доменах (Bio/Text)"),
]

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    root_dir = Path(__file__).resolve().parent
    scripts_dir = root_dir / "scripts"
    
    print("=" * 80)
    print("   ЗАПУСК ОРКЕСТРАТОРА ИССЛЕДОВАТЕЛЬСКОГО КОНВЕЙЕРА: ClustMetaLearn")
    print("=" * 80)
    
    global_start = time.time()
    env = os.environ.copy()
    src_path = str(root_dir / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    
    for filename, description in STEPS:
        script_path = scripts_dir / filename
        if not script_path.exists():
            print(f"\nФайл {filename} не найден по пути {script_path}!")
            print("Убедитесь, что скрипт лежит в папке scripts/, а run_pipeline.py в корне.")
            sys.exit(1)
            
        print(f"\n{description} ({filename})")
        start_time = time.time()
        
        result = subprocess.run([sys.executable, str(script_path)], cwd=str(root_dir), env=env)
        
        if result.returncode != 0:
            print(f"Скрипт {filename} завершился некорректно. Остановка конвейера.")
            sys.exit(result.returncode)
            
        elapsed = time.time() - start_time
        print(f"Время этапа: {elapsed:.2f} сек.")
        
    total_time = time.time() - global_start
    print("\n" + "=" * 80)
    print(f"Все этапы успешно выполнены за {total_time:.2f} сек.!")
    print("=" * 80)

if __name__ == "__main__":
    main()