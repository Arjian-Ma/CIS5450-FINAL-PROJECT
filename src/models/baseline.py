"""
src/models/baseline.py
──────────────────────
Baseline models for predicting log_copies_sold.

Models implemented (fully)
───────────────────────────
  1. MeanPredictor     – predicts the training-set mean for every sample
                         (lower-bound benchmark; any real model should beat this)
  2. LinearRegression  – OLS, no regularisation
  3. RidgeRegression   – L2-penalised linear regression
  4. LassoRegression   – L1-penalised linear regression (also does feature selection)

All models share a common interface:
    fit(X_train, y_train) → self
    predict(X)            → np.ndarray
    evaluate(X, y)        → dict  (RMSE, MAE, R² on log scale + inverse-transformed)

Usage
──────
    from src.models.baseline import run_all_baselines
    results = run_all_baselines(data_dict)   # data_dict from features.prepare_features()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge

from configs.config import LASSO_ALPHA, RANDOM_STATE, RIDGE_ALPHA
from src.evaluation.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


# ── Base wrapper ───────────────────────────────────────────────────────────────

@dataclass
class BaselineModel:
    """
    Thin wrapper that gives every baseline the same interface and stores
    train / val / test results for easy comparison.
    """
    name: str
    model: Any                          # sklearn estimator or custom
    train_metrics: dict = field(default_factory=dict)
    val_metrics:   dict = field(default_factory=dict)
    test_metrics:  dict = field(default_factory=dict)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaselineModel":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray | pd.Series,
        split_name: str = "val",
    ) -> dict:
        y_pred_log = self.predict(X)
        metrics = evaluate_predictions(np.asarray(y), y_pred_log, model_name=self.name)
        if split_name == "train":
            self.train_metrics = metrics
        elif split_name == "val":
            self.val_metrics   = metrics
        elif split_name == "test":
            self.test_metrics  = metrics
        return metrics


# ── Mean predictor ────────────────────────────────────────────────────────────

class _MeanEstimator:
    """sklearn-compatible estimator that always predicts the training mean."""

    def fit(self, X, y):
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)


def make_mean_predictor() -> BaselineModel:
    return BaselineModel(name="MeanPredictor", model=_MeanEstimator())


# ── OLS linear regression (statsmodels) ──────────────────────────────────────

class _StatsmodelsOLS:
    """
    statsmodels OLS wrapper with sklearn-compatible fit/predict interface.
    Stores the full summary (coefficients, p-values, R², F-stat, etc.)
    on self.results after fitting.
    """

    def fit(self, X, y):
        X_sm = sm.add_constant(X, has_constant="add")
        self.results = sm.OLS(y, X_sm).fit()
        return self

    def predict(self, X):
        X_sm = sm.add_constant(X, has_constant="add")
        return self.results.predict(X_sm)


def make_linear_regression() -> BaselineModel:
    return BaselineModel(
        name="LinearRegression",
        model=_StatsmodelsOLS(),
    )


def ols_summary(
    lr_model: BaselineModel,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Return a tidy DataFrame with OLS coefficients, std errors, t-stats,
    p-values, and 95 % confidence intervals for every feature.

    Parameters
    ----------
    lr_model      : fitted BaselineModel whose inner model is _StatsmodelsOLS
    feature_names : list from data["output_cols"]

    Returns
    -------
    DataFrame sorted by ascending p-value.
    """
    res = lr_model.model.results
    names = ["const"] + list(feature_names[: len(res.params) - 1])
    ci = res.conf_int()
    df = pd.DataFrame({
        "feature":   names,
        "coef":      res.params.values,
        "std_err":   res.bse.values,
        "t_stat":    res.tvalues.values,
        "p_value":   res.pvalues.values,
        "ci_low":    ci.iloc[:, 0].values,
        "ci_high":   ci.iloc[:, 1].values,
    }).sort_values("p_value").reset_index(drop=True)
    return df


# ── Ridge regression ──────────────────────────────────────────────────────────

def make_ridge(alpha: float = RIDGE_ALPHA) -> BaselineModel:
    return BaselineModel(
        name=f"Ridge(α={alpha})",
        model=Ridge(alpha=alpha, random_state=RANDOM_STATE),
    )


# ── Lasso regression ──────────────────────────────────────────────────────────

def make_lasso(alpha: float = LASSO_ALPHA) -> BaselineModel:
    return BaselineModel(
        name=f"Lasso(α={alpha})",
        model=Lasso(alpha=alpha, max_iter=5000, random_state=RANDOM_STATE),
    )


# ── Convenience runner ────────────────────────────────────────────────────────

def run_all_baselines(data: dict) -> pd.DataFrame:
    """
    Train every baseline model and evaluate on train / val / test.

    Parameters
    ----------
    data : dict returned by src.features.engineer.prepare_features()

    Returns
    -------
    summary_df : DataFrame indexed by model name, with RMSE/MAE/R² columns
                 for each split.
    """
    models = [
        make_mean_predictor(),
        make_linear_regression(),
        make_ridge(),
        make_lasso(),
    ]

    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    records = []
    for m in models:
        logger.info("Fitting %s …", m.name)
        m.fit(X_train, y_train)

        tr  = m.evaluate(X_train, y_train, "train")
        val = m.evaluate(X_val,   y_val,   "val")
        tst = m.evaluate(X_test,  y_test,  "test")

        records.append({
            "model":          m.name,
            "train_RMSE_log": tr["rmse_log"],
            "val_RMSE_log":   val["rmse_log"],
            "test_RMSE_log":  tst["rmse_log"],
            "val_MAE_log":    val["mae_log"],
            "val_R2_log":     val["r2_log"],
            "val_RMSE_raw":   val["rmse_raw"],
            "val_MAE_raw":    val["mae_raw"],
        })
        logger.info(
            "  %-30s  val RMSE(log)=%.4f  val R²=%.4f",
            m.name, val["rmse_log"], val["r2_log"],
        )

    summary = pd.DataFrame(records).set_index("model")
    return summary


# ── Lasso feature importance ───────────────────────────────────────────────────

def lasso_feature_importance(
    lasso_model: BaselineModel,
    feature_names: list[str],
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Return the non-zero Lasso coefficients sorted by absolute magnitude.
    Useful for identifying which features Lasso retained vs. zeroed out.
    """
    coef = lasso_model.model.coef_
    df = pd.DataFrame({
        "feature":   feature_names[:len(coef)],
        "coef":      coef,
        "abs_coef":  np.abs(coef),
    }).sort_values("abs_coef", ascending=False)

    nonzero = df[df["abs_coef"] > 0]
    logger.info(
        "Lasso retained %d / %d features (α=%.4f)",
        len(nonzero), len(df), lasso_model.model.alpha,
    )
    return nonzero.head(top_n).reset_index(drop=True)
