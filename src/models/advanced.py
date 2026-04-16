"""
src/models/advanced.py
──────────────────────
Framework stubs for flexible tree-based models.

Models
──────
  RandomForestModel   – ensemble of decision trees, handles non-linearity and
                        interactions (e.g. price × genre, publisher × review_ratio)
  GradientBoostModel  – sklearn GradientBoostingRegressor (fallback if XGBoost
                        is not installed)
  XGBoostModel        – XGBoost, typically the best-performing model here

All classes follow the same BaselineModel interface from src.models.baseline,
so they slot into run_all_baselines()-style evaluation loops.

Status
──────
  The class skeletons, __init__ signatures, and method stubs are complete.
  TODO markers show exactly what needs to be filled in during the modelling
  phase of the project.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from configs.config import RF_PARAMS, XGB_PARAMS, RANDOM_STATE
from src.evaluation.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared base
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdvancedModel:
    """Common interface for all advanced models."""
    name: str
    model: Any
    train_metrics: dict = field(default_factory=dict)
    val_metrics:   dict = field(default_factory=dict)
    test_metrics:  dict = field(default_factory=dict)
    feature_importances_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **fit_kwargs) -> "AdvancedModel":
        """
        Fit the model.  Subclasses may override to add early stopping or
        sample-weight logic.
        """
        # TODO: add sample weights if you want to up-weight mid-tier games
        #       (to counteract AAA games dominating MSE)
        self.model.fit(X_train, y_train, **fit_kwargs)
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_
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
            self.val_metrics = metrics
        elif split_name == "test":
            self.test_metrics = metrics
        return metrics

    def feature_importance_df(self, feature_names: list[str], top_n: int = 30) -> pd.DataFrame:
        """Return sorted feature importance DataFrame (requires model to have been fit)."""
        if self.feature_importances_ is None:
            raise RuntimeError(f"Model {self.name} has not been fitted yet.")
        df = pd.DataFrame({
            "feature":    feature_names[:len(self.feature_importances_)],
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestModel(AdvancedModel):
    """
    Random Forest regressor.

    Strengths for this dataset
    ─────────────────────────
    - Handles non-linear interactions natively (price × genre, publisher × score)
    - Robust to outliers in features
    - Built-in feature importance via mean decrease in impurity

    Known limitations
    ─────────────────
    - Does not extrapolate beyond the training range of copiesSold
    - Can over-fit on very high-cardinality encoded features if tree depth
      is unconstrained
    """

    def __init__(self, **rf_params):
        params = {**RF_PARAMS, **rf_params}
        super().__init__(
            name="RandomForest",
            model=RandomForestRegressor(**params),
        )

    # TODO (modelling phase):
    #   1. Run a grid / random search over n_estimators, max_depth,
    #      min_samples_leaf using the validation set.
    #   2. Consider max_features='sqrt' vs 'log2' for high-dim feature matrices.
    #   3. Evaluate permutation importance (not just impurity importance) to get
    #      unbiased estimates on correlated features.


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Boosting (sklearn fallback)
# ─────────────────────────────────────────────────────────────────────────────

class GradientBoostModel(AdvancedModel):
    """
    sklearn GradientBoostingRegressor.
    Slower than XGBoost but requires no extra dependency.
    Use this as a fallback or for cross-validation when XGBoost is unavailable.
    """

    def __init__(self, **gb_params):
        default_params = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )
        params = {**default_params, **gb_params}
        super().__init__(
            name="GradientBoosting",
            model=GradientBoostingRegressor(**params),
        )

    # TODO (modelling phase):
    #   1. Use staged_predict() to find the optimal n_estimators without
    #      overfitting on the validation set.
    #   2. Compare against XGBoostModel on val RMSE.


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel(AdvancedModel):
    """
    XGBoost gradient-boosted trees.

    Expected to be the best-performing model here because it can capture:
      - Price effects that differ across genres
      - Review score interactions with free-to-play status
      - Publisher reputation × follower count

    Requires:  pip install xgboost
    """

    def __init__(self, **xgb_params):
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "XGBoost is not installed. Run: pip install xgboost"
            ) from e

        params = {**XGB_PARAMS, **xgb_params}
        params.pop("random_state", None)            # XGBRegressor uses 'seed'
        params["seed"] = RANDOM_STATE
        super().__init__(name="XGBoost", model=XGBRegressor(**params))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
    ) -> "XGBoostModel":
        """
        Fit with optional early stopping on the validation set.

        Parameters
        ----------
        X_val, y_val          : Validation data for early stopping.
                                If None, no early stopping is used.
        early_stopping_rounds : Stop if val score doesn't improve for N rounds.
        """
        # TODO (modelling phase): implement early stopping
        #   fit_kwargs = {}
        #   if X_val is not None:
        #       fit_kwargs["eval_set"] = [(X_val, y_val)]
        #       fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        #       fit_kwargs["verbose"] = verbose
        #   self.model.fit(X_train, y_train, **fit_kwargs)

        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        return self

    # TODO (modelling phase):
    #   1. Tune learning_rate, max_depth, subsample, colsample_bytree via
    #      Optuna or sklearn RandomizedSearchCV.
    #   2. Try training on raw copiesSold with a custom objective (e.g. RMSLE)
    #      instead of log-transforming the target manually.
    #   3. Use SHAP values (pip install shap) for model explainability — see
    #      notebooks/03_modeling.ipynb for the planned SHAP section.


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_advanced_models(data: dict, *, use_xgboost: bool = True) -> pd.DataFrame:
    """
    Train all advanced models and return a comparison DataFrame.

    Parameters
    ----------
    data         : dict from src.features.engineer.prepare_features()
    use_xgboost  : If False, skip XGBoost (e.g. not installed in environment)

    Returns
    -------
    summary_df : DataFrame indexed by model name
    """
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    models: list[AdvancedModel] = [
        RandomForestModel(),
        GradientBoostModel(),
    ]
    if use_xgboost:
        try:
            models.append(XGBoostModel())
        except ImportError:
            logger.warning("XGBoost not available, skipping.")

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
            "  %-25s  val RMSE(log)=%.4f  val R²=%.4f",
            m.name, val["rmse_log"], val["r2_log"],
        )

    return pd.DataFrame(records).set_index("model")
