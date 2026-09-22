"""Frozen in-context evaluation on the UCI Maternal Health Risk dataset.

The current nanoMaternalPFN head is binary, so V0 evaluates:

    high risk vs low/mid risk

The pretrained PFN is frozen. Each episode supplies labeled context rows and
unlabeled query rows from the real dataset. Classical baselines are fitted only
on the same context rows used by the PFN.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from .benchmark import brier_score, expected_calibration_error
from .baselines import _binary_log_loss
from .model import NanoMaternalPFN
from .train import choose_device


UCI_DATASET_ID = 863
UCI_FEATURES: tuple[str, ...] = (
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
)


@dataclass(frozen=True)
class RealDataset:
    X: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RealResult:
    model: str
    accuracy: float
    balanced_accuracy: float
    auroc: float
    log_loss: float
    brier: float
    ece: float


def map_risk_labels(values: np.ndarray) -> np.ndarray:
    """Map UCI RiskLevel strings to high-risk binary labels."""
    normalized = np.asarray(
        [str(value).strip().lower() for value in values],
        dtype=object,
    )

    known = {"low risk", "mid risk", "high risk"}
    unknown = set(normalized.tolist()) - known
    if unknown:
        raise ValueError(f"unknown RiskLevel values: {sorted(unknown)}")

    return (normalized == "high risk").astype(np.int64)


def clean_duplicate_rows(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Remove exact duplicates and ambiguous identical feature rows.

    Identical feature vectors carrying conflicting labels are removed entirely
    so they cannot appear on opposite sides of an episode split.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    original_rows = len(y)

    combined = np.column_stack([X, y])
    _, exact_indices = np.unique(combined, axis=0, return_index=True)
    exact_indices = np.sort(exact_indices)
    X = X[exact_indices]
    y = y[exact_indices]
    after_exact_dedup = len(y)

    _, inverse = np.unique(X, axis=0, return_inverse=True)
    ambiguous_groups: set[int] = set()
    for group_id in range(int(inverse.max()) + 1):
        group_labels = np.unique(y[inverse == group_id])
        if len(group_labels) > 1:
            ambiguous_groups.add(group_id)

    if ambiguous_groups:
        keep = np.asarray(
            [group_id not in ambiguous_groups for group_id in inverse],
            dtype=bool,
        )
        X = X[keep]
        y = y[keep]

    stats = {
        "original_rows": original_rows,
        "exact_duplicates_removed": original_rows - after_exact_dedup,
        "ambiguous_rows_removed": after_exact_dedup - len(y),
        "final_rows": len(y),
    }
    return X, y, stats


def load_uci_maternal_health() -> RealDataset:
    """Fetch and clean UCI Maternal Health Risk (dataset 863)."""
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    features = dataset.data.features
    targets = dataset.data.targets

    missing = [name for name in UCI_FEATURES if name not in features.columns]
    if missing:
        raise ValueError(f"missing UCI feature columns: {missing}")

    X = features.loc[:, list(UCI_FEATURES)].to_numpy(dtype=np.float32)

    if targets.shape[1] != 1:
        raise ValueError("expected exactly one UCI target column")

    y = map_risk_labels(targets.iloc[:, 0].to_numpy())
    X, y, duplicate_stats = clean_duplicate_rows(X, y)

    if not np.isfinite(X).all():
        raise ValueError("UCI features contain non-finite values")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("binary target must contain both classes")

    return RealDataset(
        X=X,
        y=y,
        metadata={
            "dataset_id": UCI_DATASET_ID,
            "dataset_name": "UCI Maternal Health Risk",
            "target": "high risk vs low/mid risk",
            "feature_names": UCI_FEATURES,
            "high_risk_prevalence": float(y.mean()),
            **duplicate_stats,
        },
    )


def episode_indices(
    y: np.ndarray,
    *,
    n_context: int,
    n_query: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one disjoint stratified context/query episode."""
    if n_context < 2:
        raise ValueError("n_context must be at least 2")
    if n_query < 1:
        raise ValueError("n_query must be at least 1")
    if n_context + n_query > len(y):
        raise ValueError("context + query rows exceed dataset size")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_context,
        test_size=n_query,
        random_state=seed,
    )
    context_idx, query_idx = next(
        splitter.split(np.zeros((len(y), 1)), y)
    )
    return context_idx, query_idx


def _fit_logistic(X: np.ndarray, y: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=0),
    )
    model.fit(X, y)
    return model


def _fit_random_forest(X: np.ndarray, y: np.ndarray, n_trees: int):
    model = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _summarize(
    model_name: str,
    all_y: list[np.ndarray],
    all_p: list[np.ndarray],
    *,
    ece_bins: int,
) -> RealResult:
    y = np.concatenate(all_y)
    p1 = np.concatenate(all_p)
    pred = (p1 >= 0.5).astype(np.int64)

    return RealResult(
        model=model_name,
        accuracy=float((pred == y).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        auroc=float(roc_auc_score(y, p1)),
        log_loss=_binary_log_loss(y, p1),
        brier=brier_score(y, p1),
        ece=expected_calibration_error(y, p1, n_bins=ece_bins),
    )


def evaluate_uci(
    *,
    checkpoint: str,
    episodes: int = 100,
    n_context: int = 100,
    n_query: int = 50,
    seed: int = 0,
    rf_trees: int = 100,
    ece_bins: int = 10,
    device_name: str = "auto",
) -> tuple[RealDataset, list[RealResult]]:
    """Evaluate frozen PFN and task-fitted baselines on repeated real episodes."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    data = load_uci_maternal_health()
    device = choose_device(device_name)

    pfn = NanoMaternalPFN().to(device)
    pfn.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    pfn.eval()

    y_true = {
        "nanoMaternalPFN": [],
        "logistic_regression": [],
        "random_forest": [],
    }
    probabilities = {
        "nanoMaternalPFN": [],
        "logistic_regression": [],
        "random_forest": [],
    }

    with torch.no_grad():
        for episode in range(episodes):
            context_idx, query_idx = episode_indices(
                data.y,
                n_context=n_context,
                n_query=n_query,
                seed=seed + episode,
            )

            X_context = data.X[context_idx]
            y_context = data.y[context_idx]
            X_query = data.X[query_idx]
            y_query = data.y[query_idx]

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

            forest = _fit_random_forest(X_context, y_context, rf_trees)
            forest_p1 = forest.predict_proba(X_query)[:, 1]

            episode_predictions = {
                "nanoMaternalPFN": pfn_p1,
                "logistic_regression": logistic_p1,
                "random_forest": forest_p1,
            }

            for model_name, p1 in episode_predictions.items():
                y_true[model_name].append(y_query)
                probabilities[model_name].append(p1)

    results = [
        _summarize(
            model_name,
            y_true[model_name],
            probabilities[model_name],
            ece_bins=ece_bins,
        )
        for model_name in (
            "nanoMaternalPFN",
            "logistic_regression",
            "random_forest",
        )
    ]

    return data, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen nanoMaternalPFN on UCI maternal data."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--query", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    args = parser.parse_args()

    data, results = evaluate_uci(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        n_context=args.context,
        n_query=args.query,
        seed=args.seed,
        rf_trees=args.rf_trees,
        ece_bins=args.ece_bins,
        device_name=args.device,
    )

    print("UCI Maternal Health Risk")
    print("target: high risk vs low/mid risk")
    print(
        f"rows: {data.metadata['original_rows']} raw -> "
        f"{data.metadata['final_rows']} evaluation"
    )
    print(
        f"exact duplicates removed: "
        f"{data.metadata['exact_duplicates_removed']}"
    )
    print(
        f"ambiguous rows removed: "
        f"{data.metadata['ambiguous_rows_removed']}"
    )
    print(
        f"high-risk prevalence: "
        f"{data.metadata['high_risk_prevalence']:.3f}"
    )
    print(
        f"episodes: {args.episodes} | "
        f"context: {args.context} | query: {args.query}"
    )
    print()

    for result in results:
        print(result.model)
        print(
            f"  accuracy {result.accuracy:.3f} | "
            f"balanced accuracy {result.balanced_accuracy:.3f} | "
            f"AUROC {result.auroc:.3f}"
        )
        print(
            f"  log loss {result.log_loss:.4f} | "
            f"Brier {result.brier:.4f} | "
            f"ECE {result.ece:.4f}"
        )


if __name__ == "__main__":
    main()
