"""Strict frozen-transfer evaluation on the public MASS birthwt dataset."""

from __future__ import annotations

import argparse
import csv
import io
from dataclasses import dataclass
from typing import Any

import certifi
import numpy as np
import requests
import torch

from .model import NanoMaternalPFN
from .realdata import _fit_logistic, _fit_random_forest, clean_duplicate_rows
from .realdata_cv import (
    MODEL_NAMES,
    FoldMetrics,
    _mean_repeat_metrics,
    _metrics,
    sample_context_indices,
    strict_cv_splits,
    summarize_folds,
)
from .train import choose_device


BIRTHWT_CSV_URL = (
    "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/birthwt.csv"
)

# bwt is intentionally excluded because low is defined from birth weight.
BIRTHWT_FEATURES: tuple[str, ...] = (
    "age",
    "lwt",
    "race",
    "smoke",
    "ptl",
    "ht",
    "ui",
    "ftv",
)


@dataclass(frozen=True)
class BirthwtDataset:
    X: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any]


def parse_birthwt_csv(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse Rdatasets' MASS birthwt CSV without the leakage-prone bwt column."""
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []

    required = [*BIRTHWT_FEATURES, "low", "bwt"]
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(f"birthwt CSV missing columns: {missing}")

    feature_rows: list[list[float]] = []
    labels: list[int] = []

    for row_number, row in enumerate(reader, start=2):
        try:
            features = [float(row[name]) for name in BIRTHWT_FEATURES]
            label = int(row["low"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"row {row_number}: invalid numeric birthwt value"
            ) from error

        if label not in (0, 1):
            raise ValueError(
                f"row {row_number}: low must be binary 0/1, got {label}"
            )
        if not np.isfinite(features).all():
            raise ValueError(
                f"row {row_number}: predictors must be finite"
            )

        feature_rows.append(features)
        labels.append(label)

    if not feature_rows:
        raise ValueError("birthwt CSV contained no rows")

    return (
        np.asarray(feature_rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def load_birthwt() -> BirthwtDataset:
    """Download and clean the public MASS birthwt dataset."""
    response = requests.get(
        BIRTHWT_CSV_URL,
        timeout=30,
        verify=certifi.where(),
    )
    response.raise_for_status()

    raw_X, raw_y = parse_birthwt_csv(response.text)
    X, y, duplicate_stats = clean_duplicate_rows(raw_X, raw_y)

    if len(y) < 130:
        raise ValueError(
            "too few cleaned birthwt rows for the default 5-fold/100-context "
            f"protocol: {len(y)} rows"
        )

    return BirthwtDataset(
        X=X,
        y=y,
        metadata={
            "dataset_name": "MASS birthwt",
            "target": "low birth weight (<2.5 kg)",
            "feature_names": BIRTHWT_FEATURES,
            "feature_count": len(BIRTHWT_FEATURES),
            "positive_prevalence": float(y.mean()),
            **duplicate_stats,
        },
    )


def evaluate_birthwt(
    *,
    checkpoint: str,
    contexts_per_fold: int = 20,
    n_context: int = 100,
    seed: int = 0,
    rf_trees: int = 100,
    ece_bins: int = 10,
    device_name: str = "auto",
):
    """Run strict 5-fold frozen-transfer evaluation on MASS birthwt."""
    if contexts_per_fold < 1:
        raise ValueError("contexts_per_fold must be at least 1")

    data = load_birthwt()
    device = choose_device(device_name)

    pfn = NanoMaternalPFN().to(device)
    pfn.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    pfn.eval()

    results: list[FoldMetrics] = []

    with torch.no_grad():
        for fold, (train_idx, test_idx) in enumerate(
            strict_cv_splits(data.y, seed=seed),
            start=1,
        ):
            X_query = data.X[test_idx]
            y_query = data.y[test_idx]
            repeat_metrics = {
                model_name: [] for model_name in MODEL_NAMES
            }

            for repeat in range(contexts_per_fold):
                context_idx = sample_context_indices(
                    train_idx,
                    data.y,
                    n_context=n_context,
                    seed=seed + fold * 10_000 + repeat,
                )

                X_context = data.X[context_idx]
                y_context = data.y[context_idx]

                logits = pfn(
                    torch.from_numpy(X_context).unsqueeze(0).to(device),
                    torch.from_numpy(y_context).unsqueeze(0).to(device),
                    torch.from_numpy(X_query).unsqueeze(0).to(device),
                )
                pfn_p1 = (
                    torch.softmax(logits, dim=-1)[0, :, 1]
                    .float()
                    .cpu()
                    .numpy()
                )

                logistic = _fit_logistic(X_context, y_context)
                logistic_p1 = logistic.predict_proba(X_query)[:, 1]

                forest = _fit_random_forest(
                    X_context,
                    y_context,
                    rf_trees,
                )
                forest_p1 = forest.predict_proba(X_query)[:, 1]

                predictions = {
                    "nanoMaternalPFN": pfn_p1,
                    "logistic_regression": logistic_p1,
                    "random_forest": forest_p1,
                }

                for model_name, p1 in predictions.items():
                    repeat_metrics[model_name].append(
                        _metrics(y_query, p1, ece_bins=ece_bins)
                    )

            for model_name in MODEL_NAMES:
                results.append(
                    FoldMetrics(
                        fold=fold,
                        model=model_name,
                        **_mean_repeat_metrics(
                            repeat_metrics[model_name]
                        ),
                    )
                )

    metadata = {
        **data.metadata,
        "outer_folds": 5,
        "contexts_per_fold": contexts_per_fold,
        "context_rows": n_context,
    }
    return metadata, results, summarize_folds(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict frozen-transfer evaluation on MASS birthwt."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contexts-per-fold", type=int, default=20)
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    args = parser.parse_args()

    metadata, fold_results, summaries = evaluate_birthwt(
        checkpoint=args.checkpoint,
        contexts_per_fold=args.contexts_per_fold,
        n_context=args.context,
        seed=args.seed,
        rf_trees=args.rf_trees,
        ece_bins=args.ece_bins,
        device_name=args.device,
    )

    print("MASS birthwt — strict frozen transfer")
    print("target: low birth weight (<2.5 kg)")
    print(
        f"rows: {metadata['original_rows']} raw -> "
        f"{metadata['final_rows']} evaluation"
    )
    print(
        f"features: {metadata['feature_count']} "
        f"(birth weight excluded from predictors)"
    )
    print(
        f"positive prevalence: {metadata['positive_prevalence']:.3f}"
    )
    print(
        f"outer folds: 5 | contexts/fold: "
        f"{metadata['contexts_per_fold']} | "
        f"context rows: {metadata['context_rows']}"
    )
    print()

    for fold in range(1, 6):
        pfn_fold = next(
            row
            for row in fold_results
            if row.fold == fold
            and row.model == "nanoMaternalPFN"
        )
        print(
            f"fold {fold}: PFN accuracy {pfn_fold.accuracy:.3f} | "
            f"AUROC {pfn_fold.auroc:.3f}"
        )

    print()
    for summary in summaries:
        print(summary.model)
        for metric, value in summary.metrics.items():
            print(
                f"  {metric:17s} "
                f"{value.mean:.4f} ± {value.std:.4f} "
                f"(95% CI {value.ci95_low:.4f}–{value.ci95_high:.4f})"
            )
        print()


if __name__ == "__main__":
    main()
