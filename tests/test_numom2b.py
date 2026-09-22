import json

import numpy as np
import pytest

from nanomaternalpfn.numom2b import (
    load_manifest,
    load_numom2b_analysis,
)


def _manifest(csv_path, feature_columns=None):
    return {
        "dataset_name": "nuMoM2b test",
        "outcome_name": "test outcome",
        "csv_path": str(csv_path),
        "id_column": "id",
        "feature_columns": feature_columns
        or ["f1", "f2", "f3", "f4", "f5", "f6"],
        "target_column": "target",
        "positive_values": ["yes"],
        "negative_values": ["no"],
        "missing_values": ["", "NA"],
    }


def test_manifest_requires_exactly_six_features(tmp_path):
    path = tmp_path / "manifest.json"
    raw = _manifest("analysis.csv", feature_columns=["a", "b"])
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="exactly 6"):
        load_manifest(path)


def test_load_numom2b_analysis_complete_cases(tmp_path):
    csv_path = tmp_path / "analysis.csv"
    csv_path.write_text(
        "id,f1,f2,f3,f4,f5,f6,target\n"
        "p1,1,2,3,4,5,6,no\n"
        "p2,2,3,4,5,6,7,yes\n"
        "p3,3,4,5,6,7,NA,yes\n"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(csv_path)))

    data = load_numom2b_analysis(manifest_path)

    assert data.X.shape == (2, 6)
    np.testing.assert_array_equal(data.y, np.array([0, 1]))
    assert data.metadata["rows_dropped_missing"] == 1


def test_load_numom2b_analysis_rejects_duplicate_participants(tmp_path):
    csv_path = tmp_path / "analysis.csv"
    csv_path.write_text(
        "id,f1,f2,f3,f4,f5,f6,target\n"
        "p1,1,2,3,4,5,6,no\n"
        "p1,2,3,4,5,6,7,yes\n"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(csv_path)))

    with pytest.raises(ValueError, match="participant IDs must be unique"):
        load_numom2b_analysis(manifest_path)
