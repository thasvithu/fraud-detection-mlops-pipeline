from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data_ingestion import (
    EXPECTED_COLUMNS,
    load_data,
    run_data_validation,
    validate_data,
)


def _valid_df() -> pd.DataFrame:
    row = {column: 0.0 for column in EXPECTED_COLUMNS}
    row["Class"] = 0
    return pd.DataFrame([row])


def test_load_data_reads_csv(tmp_path) -> None:
    df = _valid_df()
    data_path = tmp_path / "creditcard.csv"
    df.to_csv(data_path, index=False)

    loaded = load_data(data_path)

    assert list(loaded.columns) == EXPECTED_COLUMNS
    assert loaded.shape == (1, len(EXPECTED_COLUMNS))


def test_validate_data_invalid_when_required_column_missing() -> None:
    df = _valid_df().drop(columns=["Amount"])

    report = validate_data(df)

    assert report["is_valid"] is False
    assert any("Missing required columns" in error for error in report["errors"])


def test_validate_data_invalid_when_class_has_invalid_values() -> None:
    df = _valid_df()
    df.loc[0, "Class"] = 3

    report = validate_data(df)

    assert report["is_valid"] is False
    assert any("Class contains invalid values" in error for error in report["errors"])


def test_run_data_validation_writes_report_and_fails_fast(tmp_path) -> None:
    invalid_df = _valid_df().drop(columns=["Class"])
    data_path = tmp_path / "creditcard.csv"
    report_path = tmp_path / "data_validation.json"
    invalid_df.to_csv(data_path, index=False)

    with pytest.raises(ValueError):
        run_data_validation(data_path, report_path)

    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["is_valid"] is False


def test_run_data_validation_passes_for_valid_schema(tmp_path) -> None:
    valid_df = _valid_df()
    data_path = tmp_path / "creditcard.csv"
    report_path = tmp_path / "data_validation.json"
    valid_df.to_csv(data_path, index=False)

    report = run_data_validation(data_path, report_path)

    assert report["is_valid"] is True
    assert report_path.exists()
