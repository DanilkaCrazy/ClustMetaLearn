#!/usr/bin/env python3
from pathlib import Path

from clustmetalearn.meta.evaluate import (
    EvalReport,
    cross_val_train_score,
    evaluate_classifier,
    evaluate_low_clm,
    format_report,
    topo_ablation_algrank,
    topo_ablation_cvisel,
)
from clustmetalearn.meta.models import load_bundle
from clustmetalearn.meta.train import prepare_training_table, train_meta_models

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def main():
    print("[Шаг 3] Обучение CVIsel / AlgRank и оценка meta-learning...")
    meta_csv = DATA_DIR / "meta_features.csv"
    labels_csv = DATA_DIR / "algorithm_labels.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"Нет {meta_csv}. Сначала запустите шаг 2.")

    label_arg = labels_csv if labels_csv.exists() else None
    train_meta_models(meta_csv, MODELS_DIR, labels_csv=label_arg)
    bundle = load_bundle(MODELS_DIR)
    df = prepare_training_table(meta_csv, labels_csv=label_arg)

    reports: list[EvalReport] = []
    if bundle.cvisel is not None and "target_cvi" in df.columns:
        train_labeled = df[(df["split"] == "train") & df["target_cvi"].notna()]
        cv_acc = cross_val_train_score(
            bundle.cvisel,
            df,
            feature_cols=bundle.feature_cols,
            target_col="target_cvi",
        )
        reports.append(
            EvalReport(
                task="cvisel",
                split="train_cv",
                n_samples=len(train_labeled),
                accuracy=cv_acc,
                f1_weighted=cv_acc,
            )
        )
        for split in ("val", "test"):
            if df[(df["split"] == split) & df["target_cvi"].notna()].empty:
                continue
            reports.append(
                evaluate_classifier(
                    bundle.cvisel,
                    df,
                    feature_cols=bundle.feature_cols,
                    target_col="target_cvi",
                    split_name=split,
                    task="cvisel",
                )
            )
        reports.append(
            evaluate_low_clm(
                bundle.cvisel,
                df,
                feature_cols=bundle.feature_cols,
                target_col="target_cvi",
                task="cvisel",
            )
        )

    if bundle.algrank is not None and "best_algorithm" in df.columns:
        for split in ("val", "test"):
            reports.append(
                evaluate_classifier(
                    bundle.algrank,
                    df,
                    feature_cols=bundle.feature_cols,
                    target_col="best_algorithm",
                    split_name=split,
                    task="algrank",
                )
            )

    ablation = topo_ablation_cvisel(df) if bundle.cvisel is not None else {}
    if bundle.algrank is not None:
        ablation.update({f"algo_{k}": v for k, v in topo_ablation_algrank(df).items()})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = format_report(reports, ablation=ablation)
    (REPORTS_DIR / "meta_evaluation.txt").write_text(report_text + "\n", encoding="utf-8")
    print(report_text)
    print(f"[Шаг 3 SUCCESS] Модели сохранены в {MODELS_DIR}")
    print(f"[Шаг 3 SUCCESS] Отчет сохранен в {REPORTS_DIR / 'meta_evaluation.txt'}")


if __name__ == "__main__":
    main()

