"""
src/models/pca_regression.py
────────────────────────────
Principal Component Analysis (PCA) combined with regression.
Useful for dimensionality reduction and capturing variance structure.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def fit_pca_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    n_components: int = 20,
    ridge_alpha: float = 1.0,
    random_state: int = 42,
) -> Tuple[Pipeline, float, dict]:
    """
    Fit a PCA + Ridge regression pipeline.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix
    y_train : np.ndarray
        Training target vector
    X_val : np.ndarray, optional
        Validation feature matrix for evaluation
    y_val : np.ndarray, optional
        Validation target vector
    n_components : int
        Number of principal components to retain (default 20)
    ridge_alpha : float
        Ridge regularization parameter (default 1.0)
    random_state : int
        Random seed for PCA

    Returns
    -------
    pipeline : Pipeline
        Fitted sklearn Pipeline with StandardScaler → PCA → Ridge
    train_r2 : float
        R² score on training set
    metrics_dict : dict
        Dictionary with train/val metrics and explained variance
    """
    # Build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("ridge", Ridge(alpha=ridge_alpha)),
    ])

    # Fit on training data
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_train = pipeline.predict(X_train)
    train_r2 = 1 - np.sum((y_train - y_pred_train) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))

    metrics = {
        "train_r2": train_r2,
        "train_rmse": train_rmse,
        "n_components": n_components,
        "explained_variance_ratio": pipeline.named_steps["pca"].explained_variance_ratio_,
        "cumsum_variance": np.cumsum(pipeline.named_steps["pca"].explained_variance_ratio_),
    }

    if X_val is not None and y_val is not None:
        y_pred_val = pipeline.predict(X_val)
        val_r2 = 1 - np.sum((y_val - y_pred_val) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
        val_rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
        metrics["val_r2"] = val_r2
        metrics["val_rmse"] = val_rmse

    logger.info(
        f"PCA Regression: {n_components} components, "
        f"Train R²={train_r2:.4f}, Explained Variance={metrics['cumsum_variance'][-1]:.4f}"
    )

    return pipeline, train_r2, metrics


def pca_variance_analysis(
    pca_model: PCA,
    threshold: float = 0.95,
) -> dict:
    """
    Analyze PCA variance to determine optimal number of components.

    Parameters
    ----------
    pca_model : PCA
        Fitted PCA model
    threshold : float
        Variance threshold (e.g., 0.95 for 95%) to determine n_components

    Returns
    -------
    analysis : dict
        Analysis results including optimal n_components and variance ratios
    """
    cumsum_var = np.cumsum(pca_model.explained_variance_ratio_)
    n_optimal = np.argmax(cumsum_var >= threshold) + 1

    return {
        "n_components": pca_model.n_components,
        "total_variance_explained": cumsum_var[-1],
        "n_components_for_threshold": n_optimal,
        "variance_threshold": threshold,
        "explained_variance_ratio": pca_model.explained_variance_ratio_,
        "cumsum_variance": cumsum_var,
    }


def fit_pca_by_segment(
    X_segments: dict,
    y_segments: dict,
    n_components: int = 10,
    ridge_alpha: float = 1.0,
    random_state: int = 42,
) -> dict:
    """
    Fit separate PCA + Ridge models for each market segment.

    Parameters
    ----------
    X_segments : dict
        Mapping of segment name to feature matrix
    y_segments : dict
        Mapping of segment name to target vector
    n_components : int
        Number of PCA components per segment
    ridge_alpha : float
        Ridge regularization parameter
    random_state : int
        Random seed

    Returns
    -------
    models_dict : dict
        Mapping of segment name to (pipeline, metrics)
    """
    models = {}

    for seg_name in X_segments.keys():
        X = X_segments[seg_name]
        y = y_segments[seg_name]

        if len(X) < 10:  # Skip very small segments
            logger.warning(f"Segment '{seg_name}' has only {len(X)} samples, skipping")
            continue

        # Adjust n_components if segment is too small
        n_comp_actual = min(n_components, X.shape[1], max(1, X.shape[0] // 10))

        try:
            pipeline, train_r2, metrics = fit_pca_regression(
                X, y,
                n_components=n_comp_actual,
                ridge_alpha=ridge_alpha,
                random_state=random_state,
            )
            models[seg_name] = {
                "pipeline": pipeline,
                "train_r2": train_r2,
                "metrics": metrics,
            }
            logger.info(f"Segment '{seg_name}': R²={train_r2:.4f} ({n_comp_actual} components)")
        except Exception as e:
            logger.error(f"Failed to fit PCA for segment '{seg_name}': {e}")

    return models


def compare_pca_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_components_range: list = None,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Compare performance across different numbers of PCA components.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    n_components_range : list
        Range of n_components to test. Default: [5, 10, 20, 30, 40, 50]
    ridge_alpha : float
        Ridge regularization parameter

    Returns
    -------
    results_df : pd.DataFrame
        Performance metrics for each n_components value
    """
    if n_components_range is None:
        n_components_range = [5, 10, 20, 30, 40, 50]

    records = []

    for n_comp in n_components_range:
        try:
            pipeline, train_r2, metrics = fit_pca_regression(
                X_train, y_train,
                X_val, y_val,
                n_components=n_comp,
                ridge_alpha=ridge_alpha,
            )

            record = {
                "n_components": n_comp,
                "train_r2": metrics["train_r2"],
                "train_rmse": metrics["train_rmse"],
                "val_r2": metrics.get("val_r2", np.nan),
                "val_rmse": metrics.get("val_rmse", np.nan),
                "explained_variance": metrics["cumsum_variance"][-1],
            }
            records.append(record)
        except Exception as e:
            logger.warning(f"Failed for n_components={n_comp}: {e}")

    return pd.DataFrame(records)
