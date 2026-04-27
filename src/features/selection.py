"""
src/features/selection.py
─────────────────────────
Recursive feature elimination (RFE / RFECV) for tree-based regressors.

Why this is its own module
──────────────────────────
`engineer.py` is responsible for *building* the feature matrix (imputation,
scaling, train/val/test split). Feature *selection* is a separate concern
that operates on the already-built matrix, so it lives here.

What this gives you
───────────────────
  - select_features_rfecv()  – cross-validated RFE; finds the optimal feature
                               count automatically (recommended for "best
                               performance" feature lists).
  - select_features_rfe()    – fixed-size RFE; use when you already know how
                               many features you want.
  - transform_to_selected()  – reduce a `prepare_features` data dict to the
                               selected columns so any downstream model
                               (RandomForest, GradientBoost, XGBoost) can be
                               trained on the reduced matrix without code
                               changes.
  - plot_rfecv_curve()       – save the CV-score-vs-#-features curve.
  - cv_results_table()       – per-subset-size CV RMSE table (mean, std, per fold)
                               plus the "1-SE rule" pick — the smallest feature
                               subset whose CV RMSE is within 1 standard error
                               of the best, a parsimony/performance tradeoff.

Default feature pool
────────────────────
We always include `game_age` and exclude `release_year`. The two are perfectly
collinear (game_age = 2026 - release_year), so feeding both wastes RFE rounds
and makes the ranking noisy. `game_age` is the more interpretable form.

Default estimator
─────────────────
XGBoost (gain-based importances, parallel, aligned with the strongest
downstream model). Falls back to RandomForest if xgboost isn't installed.
Pass `estimator=` to override.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import KFold

from configs.config import OUTPUT_DIR, POST_RELEASE_MODEL, RANDOM_STATE
from src.features.engineer import prepare_features

logger = logging.getLogger(__name__)

# All RFE artefacts (rankings, CV curves, logs) land here.  Keeping them in
# their own subdir prevents `outputs/` from becoming a junk drawer.
RFE_OUTPUT_DIR: Path = OUTPUT_DIR / "rfe"


# ── Default estimator used inside RFE/RFECV ───────────────────────────────────
# Why XGBoost as the default
# --------------------------
# RFE/RFECV refits the estimator across many feature subsets × CV folds. Three
# candidates were considered:
#
#   - RandomForest: fast and parallel, but importances are biased toward
#     high-cardinality continuous features and don't match how XGBoost
#     (the strongest downstream model in this project) ranks features.
#   - sklearn GradientBoostingRegressor: loss-aligned importances, but
#     single-threaded — RFECV runs 5–10× slower than XGBoost. No upside.
#   - XGBoost: parallel like RF, gain-based importances (the same metric
#     XGB uses to grow trees), and the ranking is directly aligned with the
#     downstream model.  → default choice.
#
# We keep an RF fallback for environments where xgboost isn't installed.
#
# Both defaults are tuned smaller than their production counterparts because
# the estimator is refit many times during RFE.

def _xgb_estimator():
    """XGBoost regressor sized for RFE (smaller than the production model)."""
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",     # fastest CPU algorithm
        n_jobs=-1,
        seed=RANDOM_STATE,
        verbosity=0,
    )


def _rf_estimator() -> RandomForestRegressor:
    """RandomForest fallback (used only if XGBoost is unavailable)."""
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _default_estimator():
    """Pick XGBoost when installed; otherwise fall back to RandomForest."""
    try:
        return _xgb_estimator()
    except ImportError:
        logger.warning("xgboost not installed — falling back to RandomForest for RFE.")
        return _rf_estimator()


def _build_data(df: pd.DataFrame, post_release: bool) -> dict:
    """Always pin the feature pool to {game_age yes, release_year no}."""
    return prepare_features(
        df,
        post_release=post_release,
        use_release_year=False,
        use_game_age=True,
    )


def _ranking_dataframe(
    output_cols: list[str],
    support: np.ndarray,
    ranking: np.ndarray,
    importances: np.ndarray | None,
) -> pd.DataFrame:
    """Tidy ranking table (one row per original feature)."""
    df = pd.DataFrame({
        "feature":  output_cols,
        "ranking":  ranking,         # 1 = kept, higher = eliminated earlier
        "selected": support,
    })
    if importances is not None and len(importances) == int(support.sum()):
        sel_imp = dict(zip([c for c, s in zip(output_cols, support) if s], importances))
        df["importance_in_final"] = df["feature"].map(sel_imp).fillna(0.0)
    return df.sort_values(["selected", "importance_in_final"]
                          if "importance_in_final" in df.columns else "ranking",
                          ascending=[False, False] if "importance_in_final" in df.columns else True
                          ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# RFECV — cross-validated, finds the optimal number of features
# ─────────────────────────────────────────────────────────────────────────────

def select_features_rfecv(
    df: pd.DataFrame,
    *,
    post_release: bool = POST_RELEASE_MODEL,
    estimator: Any | None = None,
    step: int = 2,
    cv: int = 3,
    min_features: int = 5,
    save: bool = True,
    save_stem: str = "rfecv_selected",
) -> dict:
    """
    Cross-validated recursive feature elimination on the training split.

    Parameters
    ----------
    df            : Pre-processed DataFrame from `02_preprocessing` (must contain `game_age`).
    post_release  : Pass-through to `prepare_features`.
    estimator     : Any sklearn-compatible estimator exposing `feature_importances_`.
                    Defaults to XGBoost (gain-based, parallel); falls back to
                    RandomForest if xgboost isn't installed.
    step          : Features eliminated per RFE round.  Larger = faster, coarser.
    cv            : Number of CV folds for scoring each feature subset.
    min_features  : Lower bound on the surviving feature count.
    save          : Persist ranking CSV + CV-score curve PNG to outputs/.
    save_stem     : Filename stem (without extension) for saved artefacts.

    Returns
    -------
    dict with keys:
      selected_features  list[str]   – names of the chosen features (RFE survivors)
      ranking            pd.DataFrame – one row per feature: ranking, selected, importance
      rfecv              RFECV        – fitted selector (use .cv_results_ for plots)
      n_optimal          int          – number of features chosen
      best_cv_rmse       float        – best mean CV RMSE (log target)
      data               dict         – the `prepare_features` output (so you can
                                        immediately call transform_to_selected())
    """
    data = _build_data(df, post_release=post_release)
    output_cols = data["output_cols"]
    X_train     = data["X_train"]
    y_train     = np.asarray(data["y_train"])

    est = estimator or _default_estimator()
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    rfecv = RFECV(
        estimator=est,
        step=step,
        cv=cv_splitter,
        scoring="neg_root_mean_squared_error",
        min_features_to_select=min_features,
        n_jobs=-1,
    )

    logger.info(
        "RFECV: starting on %d features (estimator=%s, step=%d, cv=%d)",
        X_train.shape[1], type(est).__name__, step, cv,
    )
    rfecv.fit(X_train, y_train)

    selected = [c for c, s in zip(output_cols, rfecv.support_) if s]
    importances = getattr(rfecv.estimator_, "feature_importances_", None)
    ranking_df = _ranking_dataframe(output_cols, rfecv.support_, rfecv.ranking_, importances)

    mean_scores = rfecv.cv_results_["mean_test_score"]
    best_rmse   = float(-mean_scores.max())

    logger.info(
        "RFECV: chose %d/%d features  best mean CV RMSE(log) = %.4f",
        rfecv.n_features_, len(output_cols), best_rmse,
    )

    cv_table = cv_results_table(rfecv, min_features=min_features)
    n_optimal_1se = int(cv_table.loc[cv_table["chosen_by_1se"], "n_features"].iloc[0])

    if save:
        RFE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rank_path = RFE_OUTPUT_DIR / f"{save_stem}_ranking.csv"
        ranking_df.to_csv(rank_path, index=False)
        logger.info("Saved ranking → %s", rank_path)
        cv_path = RFE_OUTPUT_DIR / f"{save_stem}_cv_table.csv"
        cv_table.to_csv(cv_path, index=False)
        logger.info("Saved CV-score table → %s", cv_path)
        plot_rfecv_curve(rfecv, save_path=RFE_OUTPUT_DIR / f"{save_stem}_curve.png", min_features=min_features)

    return {
        "selected_features": selected,
        "ranking":           ranking_df,
        "rfecv":             rfecv,
        "n_optimal":         int(rfecv.n_features_),
        "n_optimal_1se":     n_optimal_1se,
        "best_cv_rmse":      best_rmse,
        "cv_table":          cv_table,
        "data":              data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plain RFE — fixed feature count
# ─────────────────────────────────────────────────────────────────────────────

def select_features_rfe(
    df: pd.DataFrame,
    n_features_to_select: int,
    *,
    post_release: bool = POST_RELEASE_MODEL,
    estimator: Any | None = None,
    step: int = 2,
    save: bool = True,
    save_stem: str = "rfe_selected",
) -> dict:
    """
    Fixed-size recursive feature elimination.  Use when you've already decided
    how many features you want (e.g. for a plot, or after running RFECV once).
    """
    data = _build_data(df, post_release=post_release)
    output_cols = data["output_cols"]
    X_train     = data["X_train"]
    y_train     = np.asarray(data["y_train"])

    est = estimator or _default_estimator()
    rfe = RFE(estimator=est, n_features_to_select=n_features_to_select, step=step)

    logger.info(
        "RFE: starting on %d features → keep %d (estimator=%s)",
        X_train.shape[1], n_features_to_select, type(est).__name__,
    )
    rfe.fit(X_train, y_train)

    selected = [c for c, s in zip(output_cols, rfe.support_) if s]
    importances = getattr(rfe.estimator_, "feature_importances_", None)
    ranking_df = _ranking_dataframe(output_cols, rfe.support_, rfe.ranking_, importances)

    if save:
        RFE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rank_path = RFE_OUTPUT_DIR / f"{save_stem}_n{n_features_to_select}_ranking.csv"
        ranking_df.to_csv(rank_path, index=False)
        logger.info("Saved ranking → %s", rank_path)

    return {
        "selected_features": selected,
        "ranking":           ranking_df,
        "rfe":               rfe,
        "data":              data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_top_n_from_ranking(
    ranking_path: Path | str,
    n: int,
) -> list[str]:
    """
    Read the top-N feature names from a saved RFE/RFECV ranking CSV.

    Selection rule
    --------------
    1. Take RFE survivors first (selected=True), sorted by importance descending.
    2. If we still need more (n > #survivors), top up with eliminated features
       ordered by RFE `ranking` (smaller = eliminated later = better).

    Parameters
    ----------
    ranking_path : path to a ranking CSV produced by select_features_rfe(cv).
    n            : number of features to return.

    Returns
    -------
    list[str] of length min(n, total_features) — feature names ready to pass to
    `transform_to_selected`.
    """
    rank_df = pd.read_csv(ranking_path)
    if not {"feature", "ranking", "selected"}.issubset(rank_df.columns):
        raise ValueError(
            f"{ranking_path} is missing required columns "
            "(feature, ranking, selected)."
        )
    if "importance_in_final" not in rank_df.columns:
        rank_df["importance_in_final"] = 0.0

    survivors = (
        rank_df.query("selected")
               .sort_values("importance_in_final", ascending=False)
    )
    eliminated = (
        rank_df.query("not selected")
               .sort_values("ranking", ascending=True)   # smaller rank = eliminated later
    )

    if n <= len(survivors):
        return survivors.head(n)["feature"].tolist()

    pad = eliminated.head(n - len(survivors))["feature"].tolist()
    return survivors["feature"].tolist() + pad


def transform_to_selected(data: dict, selected_features: list[str]) -> dict:
    """
    Slice a `prepare_features` data dict to only the columns in `selected_features`.

    Returns a new dict shaped exactly like the original (X_train/X_val/X_test as
    ndarrays, y_* as Series, plus feature_cols / output_cols), so it drops
    straight into `run_advanced_models()` or any model class that accepts
    that contract.
    """
    output_cols = data["output_cols"]
    missing = [f for f in selected_features if f not in output_cols]
    if missing:
        raise KeyError(f"Selected features not in data['output_cols']: {missing}")

    idx = [output_cols.index(f) for f in selected_features]

    return {
        "X_train_raw": data.get("X_train_raw"),
        "X_val_raw":   data.get("X_val_raw"),
        "X_test_raw":  data.get("X_test_raw"),
        "X_train":     data["X_train"][:, idx],
        "X_val":       data["X_val"][:, idx],
        "X_test":      data["X_test"][:, idx],
        "y_train":     data["y_train"],
        "y_val":       data["y_val"],
        "y_test":      data["y_test"],
        "feature_cols": list(selected_features),
        "output_cols":  list(selected_features),
    }


def cv_results_table(
    rfecv: RFECV,
    *,
    min_features: int = 1,
    save_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Tidy per-subset-size CV results from a fitted RFECV.

    Columns
    -------
    n_features            : feature count for that subset size
    mean_cv_rmse_log      : mean CV RMSE on the log target  (lower = better)
    std_cv_rmse_log       : std-dev across folds (uncertainty band)
    split{i}_cv_rmse_log  : per-fold RMSE
    chosen_by_min         : True for the subset RFECV picked (lowest mean RMSE)
    chosen_by_1se         : True for the smallest subset within 1 std-err of the
                            best — the "1-SE rule" tradeoff between performance
                            and parsimony (common in cross-validated selection).
    """
    cv = rfecv.cv_results_
    n_subsets = len(cv["mean_test_score"])
    sizes = np.arange(min_features,
                      min_features + n_subsets * rfecv.step,
                      rfecv.step)[:n_subsets]

    df = pd.DataFrame({
        "n_features":       sizes,
        "mean_cv_rmse_log": -cv["mean_test_score"],
        "std_cv_rmse_log":  cv["std_test_score"],
    })
    for k in sorted(k for k in cv if k.startswith("split") and k.endswith("test_score")):
        df[k.replace("_test_score", "_cv_rmse_log")] = -cv[k]

    best_pos     = int(df["mean_cv_rmse_log"].idxmin())
    best_mean    = df.loc[best_pos, "mean_cv_rmse_log"]
    best_std     = df.loc[best_pos, "std_cv_rmse_log"]
    threshold    = best_mean + best_std

    one_se_pos = int(df.index[df["mean_cv_rmse_log"] <= threshold].min())

    df["chosen_by_min"] = False
    df.loc[best_pos, "chosen_by_min"] = True
    df["chosen_by_1se"] = False
    df.loc[one_se_pos, "chosen_by_1se"] = True

    df = df.sort_values("n_features").reset_index(drop=True)

    if save_path is not None:
        df.to_csv(save_path, index=False)
        logger.info("Saved CV-score table → %s", save_path)

    return df


def plot_rfecv_curve(
    rfecv: RFECV,
    *,
    save_path: Path | str | None = None,
    min_features: int = 1,
) -> None:
    """Plot mean CV RMSE (log) against the number of features kept."""
    cv_results = rfecv.cv_results_
    mean = -cv_results["mean_test_score"]
    std  = cv_results["std_test_score"]
    # The x-axis: RFECV evaluates from min_features upward in `step`-sized jumps.
    n_features = np.arange(min_features, min_features + len(mean) * rfecv.step, rfecv.step)[: len(mean)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(n_features, mean, marker="o", label="mean CV RMSE")
    ax.fill_between(n_features, mean - std, mean + std, alpha=0.2, label="±1 std")
    ax.axvline(rfecv.n_features_, color="red", linestyle="--",
               label=f"chosen = {rfecv.n_features_} features")
    ax.set_xlabel("Number of features kept")
    ax.set_ylabel("CV RMSE (log target)  — lower is better")
    ax.set_title("RFECV — feature count vs. cross-validated RMSE")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved RFECV curve → %s", save_path)
        plt.close(fig)
    else:
        plt.show()
