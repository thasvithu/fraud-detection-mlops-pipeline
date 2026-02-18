from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    SCALE_COLUMNS,
    build_preprocessor,
    handle_imbalance,
    load_preprocessor,
    preprocess_for_training,
    save_preprocessor,
    scale_features,
    split_data,
    transform_features,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 200
    fraud_count = 20

    data: dict[str, np.ndarray] = {
        "Time": rng.normal(loc=5000, scale=1000, size=rows),
        "Amount": rng.normal(loc=120, scale=50, size=rows),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=rows)

    target = np.array([0] * (rows - fraud_count) + [1] * fraud_count)
    rng.shuffle(target)
    data["Class"] = target

    return pd.DataFrame(data)


def test_split_data_is_stratified(sample_df: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.2, random_state=42)

    base_ratio = sample_df["Class"].mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    assert X_train.shape[0] == 160
    assert X_test.shape[0] == 40
    assert abs(train_ratio - base_ratio) < 0.02
    assert abs(test_ratio - base_ratio) < 0.02


def test_scale_features_transforms_only_selected_columns(sample_df: pd.DataFrame) -> None:
    features = sample_df.drop(columns=["Class"])
    scaled, scaler = scale_features(features)

    assert scaler is not None
    for column in SCALE_COLUMNS:
        assert abs(float(scaled[column].mean())) < 1e-6

    assert np.allclose(features["V1"].values, scaled["V1"].values)


def test_handle_imbalance_smote_increases_minority_class(sample_df: pd.DataFrame) -> None:
    X_train, _, y_train, _ = split_data(sample_df, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(X_train.columns.tolist())
    preprocessor.fit(X_train)
    X_train_t = transform_features(preprocessor, X_train)

    base_counts = y_train.value_counts().to_dict()
    X_balanced, y_balanced, metadata = handle_imbalance(
        X_train_t, y_train, method="smote", sampling_strategy=0.8
    )
    balanced_counts = y_balanced.value_counts().to_dict()

    assert metadata["method"] == "smote"
    assert balanced_counts[1] > base_counts[1]
    assert X_balanced.shape[0] == y_balanced.shape[0]


def test_preprocessor_save_load_roundtrip(sample_df: pd.DataFrame, tmp_path) -> None:
    X_train, _, _, _ = split_data(sample_df, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(X_train.columns.tolist())
    preprocessor.fit(X_train)

    path = tmp_path / "preprocessor.pkl"
    save_preprocessor(preprocessor, path)
    loaded = load_preprocessor(path)

    transformed = transform_features(loaded, X_train.head(5))
    assert list(transformed.columns) == X_train.columns.tolist()
    assert transformed.shape == (5, X_train.shape[1])


def test_preprocess_for_training_creates_artifact(sample_df: pd.DataFrame, tmp_path) -> None:
    artifact = tmp_path / "preprocessor.pkl"

    result = preprocess_for_training(
        sample_df,
        test_size=0.2,
        random_state=42,
        imbalance_method="class_weight",
        preprocessor_path=artifact,
    )

    assert artifact.exists()
    assert result["X_train"].shape[1] == 30
    assert result["X_test"].shape[1] == 30
    assert result["imbalance_metadata"]["method"] == "class_weight"
    assert result["imbalance_metadata"]["class_weight"] is not None


def test_handle_imbalance_rejects_unknown_method(sample_df: pd.DataFrame) -> None:
    X_train, _, y_train, _ = split_data(sample_df)
    preprocessor = build_preprocessor(X_train.columns.tolist())
    preprocessor.fit(X_train)
    X_train_t = transform_features(preprocessor, X_train)

    with pytest.raises(ValueError):
        handle_imbalance(X_train_t, y_train, method="unknown")
