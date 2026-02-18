"""Data ingestion and validation utilities for the fraud dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_ROW_COUNT = 284_807
EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
EXPECTED_CLASS_VALUES = {0, 1}


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load CSV data from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {path.suffix}")
    return pd.read_csv(path)


def get_data_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Return key dataset statistics used for validation and monitoring."""
    class_counts: dict[str, int] = {}
    fraud_ratio: float | None = None

    if "Class" in df.columns:
        raw_counts = df["Class"].value_counts(dropna=False).to_dict()
        class_counts = {str(k): int(v) for k, v in raw_counts.items()}
        if len(df) > 0:
            fraud_ratio = float((df["Class"] == 1).sum() / len(df))

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": class_counts,
        "fraud_ratio": fraud_ratio,
    }


def validate_data(df: pd.DataFrame, expected_rows: int = EXPECTED_ROW_COUNT) -> dict[str, Any]:
    """Validate schema and data quality; return a structured report."""
    errors: list[str] = []
    warnings: list[str] = []

    actual_columns = list(df.columns)
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in actual_columns]
    unexpected_columns = [col for col in actual_columns if col not in EXPECTED_COLUMNS]

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    if unexpected_columns:
        warnings.append(f"Unexpected columns present: {unexpected_columns}")

    stats = get_data_statistics(df)

    if expected_rows and stats["row_count"] != expected_rows:
        warnings.append(
            f"Row count differs from expected {expected_rows}: got {stats['row_count']}"
        )

    if stats["missing_values_total"] > 0:
        warnings.append(f"Dataset contains {stats['missing_values_total']} missing values")

    if "Class" in df.columns:
        class_values = set(df["Class"].dropna().unique().tolist())
        invalid_class_values = sorted(class_values - EXPECTED_CLASS_VALUES)
        if invalid_class_values:
            errors.append(f"Class contains invalid values: {invalid_class_values}")
        if len(class_values) == 1:
            warnings.append("Class column has only one class present")
    else:
        errors.append("Class column not found")

    is_valid = len(errors) == 0
    return {"is_valid": is_valid, "errors": errors, "warnings": warnings, "statistics": stats}


def save_validation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write validation report to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def run_data_validation(
    file_path: str | Path = "data/raw/creditcard.csv",
    report_path: str | Path = "artifacts/data_validation.json",
) -> dict[str, Any]:
    """Load dataset, validate, persist report, and fail fast on schema errors."""
    df = load_data(file_path)
    report = validate_data(df)
    save_validation_report(report, report_path)
    if not report["is_valid"]:
        raise ValueError(f"Data validation failed: {report['errors']}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate fraud dataset schema and quality.")
    parser.add_argument(
        "--data-path",
        default="data/raw/creditcard.csv",
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument(
        "--report-path",
        default="artifacts/data_validation.json",
        help="Path to write the validation report JSON.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = run_data_validation(args.data_path, args.report_path)
    print("Data validation passed.")
    print(json.dumps(report["statistics"], indent=2))


if __name__ == "__main__":
    main()
