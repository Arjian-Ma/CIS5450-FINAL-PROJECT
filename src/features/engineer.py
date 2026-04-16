"""
src/features/engineer.py
────────────────────────
Builds the final feature matrix (X) and target vector (y) from the
pre-processed DataFrame, then splits into train / validation / test sets.

Responsibilities
────────────────
  - Select the appropriate feature columns based on model framing
  - Impute remaining missing values
  - Standard-scale continuous features (fit on train, apply to val/test)
  - Produce X_train, X_val, X_test, y_train, y_val, y_test
  - Persist the column list so notebooks & main.py stay in sync
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from configs.config import (
    POST_RELEASE_FEATURES,
    POST_RELEASE_MODEL,
    RANDOM_STATE,
    TARGET_LOG,
    TARGET_RAW,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


# ── Column taxonomy ────────────────────────────────────────────────────────────
# These lists define which columns are treated as features.
# Columns not in either list are metadata (AppID, Name) or the target.

# Columns always excluded from the feature matrix regardless of framing
NON_FEATURE_COLS = {
    "AppID", "Name",
    TARGET_RAW, TARGET_LOG,
    # raw date strings that were replaced by release_year/month/quarter
    "Release date", "firstReleaseDate",
}


def get_feature_columns(
    df: pd.DataFrame,
    *,
    post_release: bool = POST_RELEASE_MODEL,
) -> list[str]:
    """
    Return the list of feature columns to use, respecting the model framing.

    In post-release mode   → all non-metadata, non-target columns are features.
    In launch-time mode    → post-release columns are additionally excluded.
    """
    excluded = set(NON_FEATURE_COLS)
    if not post_release:
        excluded.update(POST_RELEASE_FEATURES)

    feature_cols = [c for c in df.columns if c not in excluded]
    logger.info("Feature matrix: %d columns (%s mode)",
                len(feature_cols),
                "post-release" if post_release else "launch-time")
    return feature_cols


# ── Splitting ──────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_LOG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Stratified train / validation / test split.

    We stratify on `publisher_class_ord` (if present) to ensure each split
    contains a representative mix of publisher sizes — this matters because
    AAA games dominate the tail of the copiesSold distribution.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    stratify_col = "publisher_class_ord" if "publisher_class_ord" in X.columns else None

    # First split: test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=X[stratify_col] if stratify_col else None,
    )

    # Second split: validation set from remaining training data
    val_frac = VAL_SIZE / (1 - TEST_SIZE)   # adjust fraction for remaining set
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac,
        random_state=RANDOM_STATE,
        stratify=X_trainval[stratify_col] if stratify_col else None,
    )

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Preprocessing pipeline (sklearn) ──────────────────────────────────────────

def build_preprocessing_pipeline(
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str]]:
    """
    Build and fit a sklearn ColumnTransformer that:
      - Imputes missing values (median for numeric, most-frequent for binary)
      - Standard-scales all continuous columns
      - Passes binary / ordinal columns through as-is after imputation

    Parameters
    ----------
    X_train : Training feature matrix (used to fit imputers / scalers)

    Returns
    -------
    fitted_pipeline : Fitted ColumnTransformer
    output_columns  : Column names after transformation (for DataFrame reconstruction)
    """
    # Identify column types
    binary_cols = [c for c in X_train.columns
                   if X_train[c].dropna().isin([0, 1]).all()
                   and c.startswith(("genre_", "category_", "has_", "tag_",
                                     "is_free", "earlyAccess", "required_age",
                                     "has_dlc", "Windows", "Mac", "Linux"))]

    ordinal_cols = ["publisher_class_ord"] if "publisher_class_ord" in X_train.columns else []

    continuous_cols = [c for c in X_train.columns
                       if c not in binary_cols and c not in ordinal_cols]

    logger.info(
        "Pipeline columns — continuous: %d  binary: %d  ordinal: %d",
        len(continuous_cols), len(binary_cols), len(ordinal_cols),
    )

    continuous_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])

    passthrough_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    transformers = []
    if continuous_cols:
        transformers.append(("continuous", continuous_pipe, continuous_cols))
    if binary_cols or ordinal_cols:
        transformers.append(("passthrough", passthrough_pipe, binary_cols + ordinal_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    ct.fit(X_train)

    output_columns = continuous_cols + binary_cols + ordinal_cols
    return ct, output_columns


def apply_pipeline(
    ct: ColumnTransformer,
    X: pd.DataFrame,
    output_columns: list[str],
) -> np.ndarray:
    """Transform X using a fitted ColumnTransformer."""
    return ct.transform(X)


# ── Convenience wrapper ────────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_LOG,
    post_release: bool = POST_RELEASE_MODEL,
    return_pipeline: bool = False,
) -> dict:
    """
    One-stop function: selects features, splits data, fits preprocessing pipeline.

    Returns a dict with keys:
        X_train_raw, X_val_raw, X_test_raw  (DataFrames, before sklearn pipeline)
        X_train, X_val, X_test              (numpy arrays, after pipeline)
        y_train, y_val, y_test              (Series)
        feature_cols                        (list of column names)
        output_cols                         (list after ColumnTransformer)
        pipeline                            (fitted ColumnTransformer, if return_pipeline=True)
    """
    feature_cols = get_feature_columns(df, post_release=post_release)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(
        df, feature_cols, target_col=target_col
    )

    pipeline, output_cols = build_preprocessing_pipeline(X_train_raw)

    X_train = apply_pipeline(pipeline, X_train_raw, output_cols)
    X_val   = apply_pipeline(pipeline, X_val_raw,   output_cols)
    X_test  = apply_pipeline(pipeline, X_test_raw,  output_cols)

    result = dict(
        X_train_raw=X_train_raw, X_val_raw=X_val_raw, X_test_raw=X_test_raw,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_cols=feature_cols,
        output_cols=output_cols,
    )
    if return_pipeline:
        result["pipeline"] = pipeline

    return result
