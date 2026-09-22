"""Controlled-data adapter and frozen-transfer evaluation for nuMoM2b.

No DASH data are bundled with this repository. The evaluator expects a local,
one-row-per-participant analysis CSV plus an explicit JSON manifest mapping six
numeric predictor columns and one binary outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import NanoMaternalPFN
from .realdata import _fit_logistic, _fit_random_forest
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


@dataclass(frozen=True)
class NuMoM2bManifest:
    dataset_name: str
    outcome_name: str
    csv_path: str
    id_column: str
    feature_columns: tuple[str, ...]
    target_column: str
    positive_values: tuple[str, ...]
    negative_values: tuple[str, ...]
    missing_values: tuple[str, ...]


@dataclass(frozen=True)
class NuMoM2bDataset:
    participant_ids: np.ndarray
    X: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any]


def _normalized(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(value).strip().lower() for value in values)


def load_manifest(path: str | Path) -> NuMoM2bManifest:
    """Load and validate a local nuMoM2b analysis manifest."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    required = {
        "dataset_name",
        "outcome_name",
        "csv_path",
        "id_column",
        "feature_columns",
        "target_column",
        "positive_values",
        "negative_values",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"manifest is missing fields: {missing}")

    feature_columns = tuple(str(value) for value in raw["feature_columns"])
    if len(feature_columns) != 6:
        raise ValueError(
            "nanoMaternalPFN currently requires exactly 6 numeric features"
        )
    if len(set(feature_columns)) != 6:
        raise ValueError("feature_columns must be unique")

    positive_values = _normalized(raw["positive_values"])
    negative_values = _normalized(raw["negative_values"])
    if not positive_values or not negative_values:
        raise ValueError("positive_values and negative_values cannot be empty")
    if set(positive_values) & set(negative_values):
        raise ValueError("positive_values and negative_values must not overlap")

    missing_values = _normalized(
        raw.get("missing_values", ["", "na", "n/a", ".", "null"])
    )

    return NuMoM2bManifest(
        dataset_name=str(raw["dataset_name"]),
        outcome_name=str(raw["outcome_name"]),
        csv_path=str(raw["csv_path"]),
        id_column=str(raw["id_column"]),
        feature_columns=feature_columns,
        target_column=str(raw["target_column"]),
        positive_values=positive_values,
        negative_values=negative_values,
        missing_values=missing_values,
    )


def _resolve_csv_path(
    manifest_path: str | Path,
    csv_path: str,
) -> Path:
    path = Path(csv_path).expanduser()
    if path.is_absolute():
        return path

    manifest_relative = Path(manifest_path).resolve().parent / path
    if manifest_relative.exists():
        return manifest_relative

    return Path.cwd() / path


def load_numom2b_analysis(
    manifest_path: str | Path,
) -> NuMoM2bDataset:
    """Load a complete-case, one-row-per-participant local analysis CSV."""
    manifest = load_manifest(manifest_path)
    csv_path = _resolve_csv_path(manifest_path, manifest.csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"nuMoM2b analysis CSV not found: {csv_path}"
        )

    participant_ids: list[str] = []
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    dropped_missing = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required_columns = [
            manifest.id_column,
            *manifest.feature_columns,
            manifest.target_column,
        ]
        missing_columns = [
            column for column in required_columns if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"analysis CSV is missing columns: {missing_columns}"
            )

        missing_tokens = set(manifest.missing_values)
        positive = set(manifest.positive_values)
        negative = set(manifest.negative_values)

        for row_number, row in enumerate(reader, start=2):
            raw_id = str(row[manifest.id_column]).strip()
            raw_target = str(row[manifest.target_column]).strip()
            raw_features = [
                str(row[column]).strip()
                for column in manifest.feature_columns
            ]

            normalized_values = [
                raw_id.lower(),
                raw_target.lower(),
                *[value.lower() for value in raw_features],
            ]
            if any(value in missing_tokens for value in normalized_values):
                dropped_missing += 1
                continue

            normalized_target = raw_target.lower()
            if normalized_target in positive:
                label = 1
            elif normalized_target in negative:
                label = 0
            else:
                raise ValueError(
                    f"row {row_number}: target value {raw_target!r} "
                    "is not listed in positive_values/negative_values"
                )

            try:
                features = [float(value) for value in raw_features]
            except ValueError as error:
                raise ValueError(
                    f"row {row_number}: all six predictor values must "
                    "be numeric"
                ) from error

            if not np.isfinite(features).all():
                raise ValueError(
                    f"row {row_number}: predictor values must be finite"
                )

            participant_ids.append(raw_id)
            feature_rows.append(features)
            labels.append(label)

    if not feature_rows:
        raise ValueError("no complete analysis rows remain")

    if len(set(participant_ids)) != len(participant_ids):
        raise ValueError(
            "participant IDs must be unique; construct one analysis row "
            "per participant before evaluation"
        )

    X = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    ids = np.asarray(participant_ids, dtype=object)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("analysis dataset must contain both outcome classes")

    return NuMoM2bDataset(
        participant_ids=ids,
        X=X,
        y=y,
        metadata={
            "dataset_name": manifest.dataset_name,
            "outcome_name": manifest.outcome_name,
            "feature_columns": manifest.feature_columns,
            "target_column": manifest.target_column,
            "analysis_csv": str(csv_path),
            "rows": len(y),
            "rows_dropped_missing": dropped_missing,
            "positive_prevalence": float(y.mean()),
        },
    )


def inspect_csv(path: str | Path) -> None:
    """Print local CSV columns without loading protected data into git."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        row_count = sum(1 for _ in reader)

    print(f"rows: {row_count}")
    print(f"columns: {len(header)}")
    for column in header:
        print(column)


def evaluate_numom2b(
    *,
    manifest_path: str,
    checkpoint: str,
    contexts_per_fold: int = 20,
    n_context: int = 100,
    seed: int = 0,
    rf_trees: int = 100,
    ece_bins: int = 10,
    device_name: str = "auto",
):
    """Run strict five-fold frozen-transfer evaluation on local nuMoM2b data."""
    data = load_numom2b_analysis(manifest_path)
    if contexts_per_fold < 1:
        raise ValueError("contexts_per_fold must be at least 1")

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local controlled-data adapter for nuMoM2b."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="List columns in a local DASH-derived CSV.",
    )
    inspect_parser.add_argument("--csv", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run strict frozen-transfer evaluation.",
    )
    evaluate_parser.add_argument("--manifest", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--contexts-per-fold", type=int, default=20)
    evaluate_parser.add_argument("--context", type=int, default=100)
    evaluate_parser.add_argument("--seed", type=int, default=0)
    evaluate_parser.add_argument("--rf-trees", type=int, default=100)
    evaluate_parser.add_argument("--ece-bins", type=int, default=10)
    evaluate_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "inspect":
        inspect_csv(args.csv)
        return

    metadata, fold_results, summaries = evaluate_numom2b(
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        contexts_per_fold=args.contexts_per_fold,
        n_context=args.context,
        seed=args.seed,
        rf_trees=args.rf_trees,
        ece_bins=args.ece_bins,
        device_name=args.device,
    )

    print(f"{metadata['dataset_name']} — strict frozen transfer")
    print(f"outcome: {metadata['outcome_name']}")
    print(
        f"rows: {metadata['rows']} | "
        f"dropped missing: {metadata['rows_dropped_missing']} | "
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
