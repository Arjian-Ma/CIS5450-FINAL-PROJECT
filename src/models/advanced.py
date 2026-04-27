"""
src/models/advanced.py
──────────────────────
Tree-based models for Steam sales prediction.

Models
──────
  RandomForestModel   – ensemble of decision trees; OOB score as quality proxy
  GradientBoostModel  – sklearn GradientBoostingRegressor with staged loss curve
  XGBoostModel        – XGBoost with early stopping and full loss curve
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from configs.config import RF_PARAMS, XGB_PARAMS, RANDOM_STATE, MODELS_DIR
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
    target_scale: str = "log"   # "log" if trained on log1p target, "raw" if on copiesSold
    train_metrics: dict = field(default_factory=dict)
    val_metrics:   dict = field(default_factory=dict)
    test_metrics:  dict = field(default_factory=dict)
    feature_importances_: np.ndarray | None = field(default=None, repr=False)
    loss_curve_: dict | None = field(default=None, repr=False)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **fit_kwargs) -> "AdvancedModel":
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
        y_pred = self.predict(X)
        metrics = evaluate_predictions(
            np.asarray(y), y_pred,
            target_scale=self.target_scale,
            model_name=self.name,
        )
        if split_name == "train":
            self.train_metrics = metrics
        elif split_name == "val":
            self.val_metrics = metrics
        elif split_name == "test":
            self.test_metrics = metrics
        return metrics

    def feature_importance_df(self, feature_names: list[str], top_n: int = 30) -> pd.DataFrame:
        if self.feature_importances_ is None:
            raise RuntimeError(f"Model {self.name} has not been fitted yet.")
        df = pd.DataFrame({
            "feature":    feature_names[:len(self.feature_importances_)],
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_n)

    def save(self, path=None) -> str:
        """Save model to models/ directory using joblib. Returns the saved path."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = path or MODELS_DIR / f"{self.name}.joblib"
        joblib.dump(self, save_path)
        print(f"[{self.name}] saved → {save_path}")
        return str(save_path)

    @classmethod
    def load(cls, path) -> "AdvancedModel":
        """Load a saved model from disk."""
        return joblib.load(path)

    def plot_loss_curve(self, ax=None) -> None:
        """Plot train/val loss curve if available for this model."""
        if self.loss_curve_ is None:
            print(f"{self.name}: no loss curve available.")
            return
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        if "train" in self.loss_curve_:
            ax.plot(self.loss_curve_["train"], label="train RMSE")
        if "val" in self.loss_curve_:
            ax.plot(self.loss_curve_["val"], label="val RMSE")
        if "best_iteration" in self.loss_curve_:
            ax.axvline(self.loss_curve_["best_iteration"], color="red",
                       linestyle="--", label=f"best iter={self.loss_curve_['best_iteration']}")
        ax.set_xlabel("Iteration / tree")
        ax.set_ylabel("RMSE (log scale)")
        ax.set_title(f"{self.name} — loss curve")
        ax.legend()
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestModel(AdvancedModel):
    """
    Random Forest regressor.

    Loss curve: not applicable — trees are built independently so there is no
    iterative loss to track. OOB (out-of-bag) RMSE is stored in oob_rmse_ as
    the closest quality proxy.
    """

    def __init__(self, **rf_params):
        params = {**RF_PARAMS, **rf_params}
        params.setdefault("oob_score", True)   # enables oob_score_ after fit
        super().__init__(
            name="RandomForest",
            model=RandomForestRegressor(**params),
        )
        self.oob_rmse_: float | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **fit_kwargs) -> "RandomForestModel":
        super().fit(X_train, y_train, **fit_kwargs)
        if hasattr(self.model, "oob_score_"):
            # oob_score_ is R², convert to RMSE via residuals
            y_oob = self.model.oob_prediction_
            self.oob_rmse_ = float(root_mean_squared_error(y_train, y_oob))
            logger.info("RandomForest OOB RMSE (log): %.4f", self.oob_rmse_)
        return self

    def plot_loss_curve(self, ax=None) -> None:
        if self.oob_rmse_ is None:
            print("RandomForest: fit the model first.")
            return
        print(
            f"RandomForest has no iterative loss curve.\n"
            f"OOB RMSE (log scale): {self.oob_rmse_:.4f}\n"
            f"OOB R²: {self.model.oob_score_:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Boosting (sklearn)
# ─────────────────────────────────────────────────────────────────────────────

class GradientBoostModel(AdvancedModel):
    """
    sklearn GradientBoostingRegressor with per-iteration train/val loss curve
    computed via staged_predict().
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

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **fit_kwargs,
    ) -> "GradientBoostModel":
        self.model.fit(X_train, y_train, **fit_kwargs)
        self.feature_importances_ = self.model.feature_importances_

        # Build loss curve using staged_predict
        train_rmse, val_rmse = [], []
        for y_pred_train in self.model.staged_predict(X_train):
            train_rmse.append(root_mean_squared_error(y_train, y_pred_train))
        if X_val is not None and y_val is not None:
            for y_pred_val in self.model.staged_predict(X_val):
                val_rmse.append(root_mean_squared_error(y_val, y_pred_val))

        self.loss_curve_ = {"train": train_rmse}
        if val_rmse:
            self.loss_curve_["val"] = val_rmse
            best = int(np.argmin(val_rmse))
            self.loss_curve_["best_iteration"] = best
            logger.info("GradientBoosting best val RMSE %.4f at iteration %d", val_rmse[best], best)

        return self


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel(AdvancedModel):
    """
    XGBoost gradient-boosted trees with early stopping and full loss curve.
    Requires:  pip install xgboost
    """

    def __init__(self, early_stopping_rounds: int = 50, **xgb_params):
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost") from e

        params = {**XGB_PARAMS, **xgb_params}
        params.pop("random_state", None)
        params["seed"] = RANDOM_STATE
        self._early_stopping_rounds = early_stopping_rounds
        super().__init__(name="XGBoost", model=XGBRegressor(**params))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> "XGBoostModel":
        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            self.model.early_stopping_rounds = self._early_stopping_rounds
            fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]
            fit_kwargs["verbose"] = verbose
        else:
            self.model.early_stopping_rounds = None

        self.model.fit(X_train, y_train, **fit_kwargs)
        self.feature_importances_ = self.model.feature_importances_

        # Store loss curve from evals_result_ (only available when eval_set was passed)
        evals = self.model.evals_result() if fit_kwargs.get("eval_set") else {}
        if evals:
            metric = list(list(evals.values())[0].keys())[0]
            sets   = list(evals.keys())
            self.loss_curve_ = {
                "train": evals[sets[0]][metric],
            }
            if len(sets) > 1:
                self.loss_curve_["val"] = evals[sets[1]][metric]
                best = int(self.model.best_iteration)
                self.loss_curve_["best_iteration"] = best
                logger.info("XGBoost early stopped at iteration %d", best)

        return self


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_advanced_models(
    data: dict, *, use_xgboost: bool = True, save: bool = False
) -> tuple[pd.DataFrame, dict[str, AdvancedModel]]:
    """
    Train all advanced models and return a comparison DataFrame and fitted models.

    Returns
    -------
    summary  : DataFrame of metrics per model
    models   : dict mapping model name → fitted model object
               e.g. models["XGBoost"].plot_loss_curve()
                    models["RandomForest"].feature_importance_df(feature_names)
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
        print(f"[{m.name}] training started …")
        if isinstance(m, (GradientBoostModel, XGBoostModel)):
            m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            m.fit(X_train, y_train)

        tr  = m.evaluate(X_train, y_train, "train")
        val = m.evaluate(X_val,   y_val,   "val")
        tst = m.evaluate(X_test,  y_test,  "test")

        print(f"[{m.name}] done — val RMSE(log)={val['rmse_log']:.4f}  val R²={val['r2_log']:.4f}")
        if save:
            m.save()

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

    summary = pd.DataFrame(records).set_index("model")
    fitted  = {m.name: m for m in models}
    return summary, fitted
