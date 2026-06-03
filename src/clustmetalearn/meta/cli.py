"""CLI for meta-learning pipeline."""

from __future__ import annotations

import argparse
import json
import sys
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
from clustmetalearn.meta.labels import build_labels_for_datasets_root
from clustmetalearn.meta.models import load_bundle
from clustmetalearn.meta.recommend import recommend_from_table
from clustmetalearn.meta.train import prepare_training_table, train_meta_models


def _default_meta_csv() -> Path:
    candidates = [
        Path("Этап 2. Обучение мета-модели для предсказания CVI и анализ ISA/datasets/meta_features_gpu.csv"),
        Path("datasets/meta_features_gpu.csv"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def cmd_build_labels(args: argparse.Namespace) -> None:
    summary = Path(args.summary) if args.summary else None
    build_labels_for_datasets_root(
        Path(args.datasets_root),
        Path(args.output),
        summary_path=summary,
    )
    print("Saved:", args.output)


def cmd_train(args: argparse.Namespace) -> None:
    labels = Path(args.labels) if args.labels else None
    bundle = train_meta_models(
        Path(args.meta_csv),
        Path(args.models_dir),
        labels_csv=labels,
        use_topo=not args.no_topo,
        random_state=args.seed,
    )
    print("Models saved to:", args.models_dir)
    print("  CVIsel:", bundle.cvisel is not None)
    print("  AlgRank:", bundle.algrank is not None)
    print("  Features:", len(bundle.feature_cols))


def cmd_evaluate(args: argparse.Namespace) -> None:
    df = prepare_training_table(
        Path(args.meta_csv),
        labels_csv=Path(args.labels) if args.labels else None,
        use_topo=not args.no_topo,
    )
    bundle = load_bundle(Path(args.models_dir))
    reports = []

    if bundle.cvisel is not None and "target_cvi" in df.columns:
        train_labeled = df[(df["split"] == "train") & df["target_cvi"].notna()]
        if len(train_labeled) > 0:
            cv_acc = cross_val_train_score(
                bundle.cvisel,
                df,
                feature_cols=bundle.feature_cols,
                target_col="target_cvi",
            )
            print(f"CVIsel train CV accuracy (3-fold): {cv_acc:.3f}")
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
            labeled = df[(df["split"] == split) & df["target_cvi"].notna()]
            if len(labeled) == 0:
                print(f"CVIsel: skip split={split} (no labels in meta table).")
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
                    top_k=3,
                )
            )
        reports.append(
            evaluate_low_clm(
                bundle.algrank,
                df,
                feature_cols=bundle.feature_cols,
                target_col="best_algorithm",
                task="algrank",
            )
        )

    ablation = topo_ablation_cvisel(df) if not args.no_topo else None
    if "best_algorithm" in df.columns:
        ab_algo = topo_ablation_algrank(df)
        if ablation is None:
            ablation = ab_algo
        elif ab_algo:
            ablation = {**ablation, **{f"algo_{k}": v for k, v in ab_algo.items()}}

    text = format_report(reports, ablation=ablation)
    print(text)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")
        print("Report:", args.report)


def cmd_recommend(args: argparse.Namespace) -> None:
    rec = recommend_from_table(
        args.csv_path,
        Path(args.models_dir),
        label_column=args.label_column,
        include_topology=not args.no_topo,
    )
    out = {
        "predicted_cvi": rec.predicted_cvi,
        "predicted_cvi_metric": rec.predicted_cvi_metric,
        "predicted_algorithm": rec.predicted_algorithm,
        "top3_algorithms": list(rec.top3_algorithms),
        "hp_intervals": rec.hp_intervals,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ClustMetaLearn meta-learning CLI")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-labels", help="Grid-search best algorithm per .bin dataset")
    b.add_argument("datasets_root", help="Root of hj-n/labeled-datasets clone")
    b.add_argument("-o", "--output", default="data/algorithm_labels.csv")
    b.add_argument("--summary", default=None, help="dataset_summary.csv for shapes")
    b.set_defaults(func=cmd_build_labels)

    t = sub.add_parser("train", help="Train CVIsel and AlgRank")
    t.add_argument("--meta-csv", default=str(_default_meta_csv()))
    t.add_argument("--labels", default=None, help="algorithm_labels.csv from build-labels")
    t.add_argument("--models-dir", default="models")
    t.add_argument("--no-topo", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="Metrics, low-CLM, topo ablation")
    e.add_argument("--meta-csv", default=str(_default_meta_csv()))
    e.add_argument("--labels", default=None)
    e.add_argument("--models-dir", default="models")
    e.add_argument("--no-topo", action="store_true")
    e.add_argument("--report", default=None)
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("recommend", help="Recommend CVI and algorithm for a CSV")
    r.add_argument("csv_path")
    r.add_argument("--models-dir", default="models")
    r.add_argument("-l", "--label-column", default=None)
    r.add_argument("--no-topo", action="store_true")
    r.set_defaults(func=cmd_recommend)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
