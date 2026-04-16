"""
src/evaluation/metrics.py
─────────────────────────
Evaluation utilities for regression predictions on log-transformed targets.

Because copiesSold is log1p-transformed before modelling, we report metrics
on both:
  - the log scale      (RMSE_log, MAE_log, R²_log)  — primary for comparing models
  - the raw (copies)   (RMSE_raw, MAE_raw)            — for business interpretation

The raw-scale errors are computed by applying expm1() to predictions and
actuals before calculating the metric.  Beware that a few blockbuster games
will dominate raw-scale RMSE — that's why log-scale metrics are primary.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ── Core metric computation ────────────────────────────────────────────────────

def evaluate_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    *,
    model_name: str = "",
) -> dict:
    """
    Compute RMSE, MAE, and R² on both the log scale and the original copies-sold
    scale.

    Parameters
    ----------
    y_true_log : Ground-truth values in log1p space.
    y_pred_log : Predicted values in log1p space.
    model_name : Optional label (used in log messages only).

    Returns
    -------
    dict with keys:
        rmse_log, mae_log, r2_log
        rmse_raw, mae_raw
        (No R² on raw scale: MSE is dominated by blockbuster outliers and the
         raw R² is not a meaningful comparator across differently-scaled models.)
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    # Log-scale metrics
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae_log  = float(mean_absolute_error(y_true_log, y_pred_log))
    r2_log   = float(r2_score(y_true_log, y_pred_log))

    # Raw-scale metrics (inverse transform)
    y_true_raw = np.expm1(y_true_log)
    y_pred_raw = np.expm1(y_pred_log)
    rmse_raw   = float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
    mae_raw    = float(mean_absolute_error(y_true_raw, y_pred_raw))

    metrics = dict(
        rmse_log=rmse_log,
        mae_log=mae_log,
        r2_log=r2_log,
        rmse_raw=rmse_raw,
        mae_raw=mae_raw,
    )

    if model_name:
        logger.debug(
            "[%s]  RMSE_log=%.4f  MAE_log=%.4f  R²_log=%.4f  "
            "RMSE_raw=%.0f  MAE_raw=%.0f",
            model_name, rmse_log, mae_log, r2_log, rmse_raw, mae_raw,
        )

    return metrics


# ── Comparison table ──────────────────────────────────────────────────────────

def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a single comparison DataFrame from a {model_name: metrics_dict} map.

    Example
    ──────
        from src.evaluation.metrics import compare_models
        table = compare_models({
            "MeanPredictor": mean_val_metrics,
            "Ridge":         ridge_val_metrics,
            "XGBoost":       xgb_val_metrics,
        })
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "model":       name,
            "RMSE_log":    round(m.get("rmse_log", np.nan), 4),
            "MAE_log":     round(m.get("mae_log",  np.nan), 4),
            "R²_log":      round(m.get("r2_log",   np.nan), 4),
            "RMSE_raw":    round(m.get("rmse_raw", np.nan), 0),
            "MAE_raw":     round(m.get("mae_raw",  np.nan), 0),
        })
    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("RMSE_log")
    return df


# ── Residual analysis helpers ─────────────────────────────────────────────────

def residual_df(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of residuals for diagnostic plots.

    Columns: y_true_log, y_pred_log, residual, abs_residual,
             y_true_raw, y_pred_raw   (and any columns from metadata)

    metadata, if provided, should be a DataFrame aligned with the prediction
    arrays (e.g. X_test_raw with AppID, Name, publisherClass).
    """
    df = pd.DataFrame({
        "y_true_log":  y_true_log,
        "y_pred_log":  y_pred_log,
        "residual":    y_pred_log - y_true_log,
        "abs_residual": np.abs(y_pred_log - y_true_log),
        "y_true_raw":  np.expm1(y_true_log),
        "y_pred_raw":  np.expm1(y_pred_log),
    })
    if metadata is not None:
        df = pd.concat([df.reset_index(drop=True),
                        metadata.reset_index(drop=True)], axis=1)
    return df


def quantile_rmse(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    quantiles: Sequence[float] = (0.25, 0.50, 0.75, 0.90, 0.95),
) -> pd.DataFrame:
    """
    Compute RMSE separately for games above/below percentile breakpoints of
    the actual copiesSold distribution.

    Helps diagnose whether the model is better at predicting low-selling or
    high-selling games — relevant because blockbusters dominate raw RMSE.
    """
    y_true = np.asarray(y_true_log)
    y_pred = np.asarray(y_pred_log)

    records = []
    thresholds = [np.quantile(y_true, q) for q in quantiles]

    for q, thresh in zip(quantiles, thresholds):
        mask_low  = y_true <= thresh
        mask_high = y_true >  thresh

        rmse_low  = float(np.sqrt(mean_squared_error(y_true[mask_low],  y_pred[mask_low])))  if mask_low.any()  else np.nan
        rmse_high = float(np.sqrt(mean_squared_error(y_true[mask_high], y_pred[mask_high]))) if mask_high.any() else np.nan

        records.append({
            "percentile":      f"≤ {int(q*100)}th",
            "n_games":         int(mask_low.sum()),
            "RMSE_log_below":  round(rmse_low,  4),
            "RMSE_log_above":  round(rmse_high, 4),
        })

    return pd.DataFrame(records)


# ── Predicted vs actual plot ───────────────────────────────────────────────────

def plot_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    model_name: str = "Model",
    sample_n: int = 3000,
) -> None:
    """
    Plot predicted vs actual copiesSold on both log and raw scales.

    Parameters
    ----------
    y_true_log : Ground-truth log1p(copiesSold) from the test set.
    y_pred_log : Predicted log1p(copiesSold).
    model_name : Label shown in plot titles.
    sample_n   : Max points to scatter (random sample to avoid overplotting).
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    # Random subsample for scatter readability
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true_log), size=min(sample_n, len(y_true_log)), replace=False)
    yt_log, yp_log = y_true_log[idx], y_pred_log[idx]
    yt_raw, yp_raw = np.expm1(yt_log), np.expm1(yp_log)

    r2  = r2_score(y_true_log, y_pred_log)
    rmse = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{model_name} — Predicted vs Actual (test set)  |  R²={r2:.3f}  RMSE_log={rmse:.3f}")

    # ── Panel 1: log scale scatter ──
    lims = [min(yt_log.min(), yp_log.min()) - 0.5,
            max(yt_log.max(), yp_log.max()) + 0.5]
    axes[0].scatter(yt_log, yp_log, alpha=0.2, s=8, color="steelblue")
    axes[0].plot(lims, lims, "r--", linewidth=1, label="perfect")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("Actual log1p(copiesSold)")
    axes[0].set_ylabel("Predicted log1p(copiesSold)")
    axes[0].set_title("Log scale")
    axes[0].legend()

    # ── Panel 2: raw scale scatter (log-log axes for readability) ──
    mask = (yt_raw > 0) & (yp_raw > 0)
    axes[1].scatter(yt_raw[mask], yp_raw[mask], alpha=0.2, s=8, color="darkorange")
    raw_lims = [1, max(yt_raw.max(), yp_raw.max()) * 1.1]
    axes[1].plot(raw_lims, raw_lims, "r--", linewidth=1, label="perfect")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlim(raw_lims); axes[1].set_ylim(raw_lims)
    axes[1].set_xlabel("Actual copiesSold")
    axes[1].set_ylabel("Predicted copiesSold")
    axes[1].set_title("Raw scale (log-log axes)")
    axes[1].legend()

    # ── Panel 3: residual histogram ──
    residuals = yp_log - yt_log
    axes[2].hist(residuals, bins=60, edgecolor="none", color="mediumseagreen")
    axes[2].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Residual (predicted − actual, log scale)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Residual distribution  (mean={residuals.mean():.3f})")

    plt.tight_layout()
    plt.show()


def plot_sample_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    model_name: str = "Model",
    n: int = 20,
    labels: list[str] | None = None,
) -> None:
    """
    Bar chart of predicted vs actual copiesSold for n random games from the test set.

    Parameters
    ----------
    y_true_log : Ground-truth log1p(copiesSold).
    y_pred_log : Predicted log1p(copiesSold).
    model_name : Label for the title.
    n          : Number of games to sample.
    labels     : Optional game name list aligned with y_true_log (e.g. df_test["Name"].values).
                 If None, games are labelled by index.
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true_log), size=min(n, len(y_true_log)), replace=False)
    idx = idx[np.argsort(y_true_log[idx])]  # sort by actual so chart reads cleanly

    actual    = np.expm1(y_true_log[idx])
    predicted = np.expm1(y_pred_log[idx])
    game_labels = [labels[i] for i in idx] if labels is not None else [str(i) for i in idx]

    x = np.arange(len(idx))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, actual,    width, label="Actual",    color="steelblue")
    ax.bar(x + width / 2, predicted, width, label="Predicted", color="darkorange", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(game_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("copiesSold (log scale)")
    ax.set_title(f"{model_name} — Predicted vs Actual for {len(idx)} sampled games")
    ax.legend()
    plt.tight_layout()
    plt.show()
