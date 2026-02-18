"""Training/inference preprocessing pipeline utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

TARGET_COLUMN = "Class"
SCALE_COLUMNS = ["Time", "Amount"]


def split_data(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train/test with class stratification."""
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_features(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale selected columns and return transformed dataframe and scaler."""
    scale_columns = columns or SCALE_COLUMNS
    missing = [column for column in scale_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Columns not found for scaling: {missing}")

    local_scaler = scaler or StandardScaler()
    result = df.copy()
    result[scale_columns] = local_scaler.fit_transform(df[scale_columns])
    return result, local_scaler


def build_preprocessor(
    feature_columns: list[str],
    *,
    scale_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Build a column transformer for consistent training/inference transforms."""
    chosen_scale_columns = scale_columns or SCALE_COLUMNS
    missing = [column for column in chosen_scale_columns if column not in feature_columns]
    if missing:
        raise ValueError(f"Scale columns missing from features: {missing}")

    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), chosen_scale_columns)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Transform feature dataframe using a fitted preprocessor."""
    transformed = preprocessor.transform(X)
    if not isinstance(transformed, pd.DataFrame):
        transformed = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())
    return transformed


def handle_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    method: str = "class_weight",
    random_state: int = 42,
    sampling_strategy: float = 0.5,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Handle class imbalance using strategy selected by method."""
    selected = method.lower()
    if selected not in {"none", "class_weight", "smote"}:
        raise ValueError("method must be one of: none, class_weight, smote")

    if selected == "none":
        return X_train, y_train, {"method": "none", "class_weight": None}

    if selected == "class_weight":
        classes = np.array(sorted(y_train.unique().tolist()))
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}
        return X_train, y_train, {"method": "class_weight", "class_weight": class_weight}

    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    X_balanced = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_balanced = pd.Series(y_resampled, name=y_train.name)
    return X_balanced, y_balanced, {"method": "smote", "class_weight": None}


def save_preprocessor(preprocessor: ColumnTransformer, output_path: str | Path) -> Path:
    """Persist fitted preprocessor to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    return path


def load_preprocessor(preprocessor_path: str | Path) -> ColumnTransformer:
    """Load persisted preprocessor from disk."""
    return joblib.load(Path(preprocessor_path))


def preprocess_for_training(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
    imbalance_method: str = "class_weight",
    preprocessor_path: str | Path = "models/preprocessor.pkl",
) -> dict[str, Any]:
    """Run train/test split, fit/transform preprocessor, and handle imbalance."""
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(feature_columns=X_train_raw.columns.tolist())
    preprocessor.fit(X_train_raw)

    X_train = transform_features(preprocessor, X_train_raw)
    X_test = transform_features(preprocessor, X_test_raw)

    X_train_final, y_train_final, imbalance_metadata = handle_imbalance(
        X_train,
        y_train,
        method=imbalance_method,
        random_state=random_state,
    )

    save_preprocessor(preprocessor, preprocessor_path)

    return {
        "X_train": X_train_final,
        "X_test": X_test,
        "y_train": y_train_final,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "imbalance_metadata": imbalance_metadata,
    }
