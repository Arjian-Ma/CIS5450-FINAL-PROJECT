"""
src/models/nn_grid_search.py
────────────────────────────
Grid search over learning rate for a fixed 5-layer MLP with dropout=0.3.

Architecture (fixed)
────────────────────
    input → 512 → 256 → 128 → 64 → 32 → 1
    BatchNorm=True   Dropout=0.3   max_epochs=150   patience=15

Learning rates searched
───────────────────────
    [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]

Usage
─────
    from src.models.nn_grid_search import run_lr_grid_search

    results_df, best_model, best_lr = run_lr_grid_search(
        data,          # dict from prepare_features()
        verbose=True,
    )
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.nn import NeuralNetModel

logger = logging.getLogger(__name__)

# ── Grid ──────────────────────────────────────────────────────────────────────

LR_GRID: List[float] = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]

# Fixed architecture — 5-layer MLP with dropout=0.3
BASE_CONFIG: dict = dict(
    hidden_layers=[512, 256, 128, 64, 32],
    dropout_rate=0.3,
    batch_norm=True,
    batch_size=256,
    max_epochs=150,
    patience=15,
)


# ── Core grid search ──────────────────────────────────────────────────────────

def run_lr_grid_search(
    data: dict,
    lr_grid: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, NeuralNetModel, float]:
    """
    Grid search over learning rate for the fixed 5-layer MLP (dropout=0.3).

    Parameters
    ----------
    data     : dict from src.features.engineer.prepare_features()
               Must contain X_train, X_val, X_test, y_train, y_val, y_test.
    lr_grid  : List of learning rates to try. Defaults to LR_GRID.
    verbose  : Print training progress.

    Returns
    -------
    results_df  : DataFrame with one row per lr — train/val/test metrics.
    best_model  : Fitted NeuralNetModel with lowest val RMSE.
    best_lr     : The learning rate of the best model.
    """
    if lr_grid is None:
        lr_grid = LR_GRID

    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]
    y_train = np.asarray(data["y_train"])
    y_val   = np.asarray(data["y_val"])
    y_test  = np.asarray(data["y_test"])

    records:    List[dict]         = []
    all_models: dict[float, NeuralNetModel] = {}

    for lr in lr_grid:
        name = f"NN5_lr{lr:.0e}"
        logger.info("=" * 60)
        logger.info("Grid search — lr=%.0e  dropout=0.3", lr)
        logger.info("=" * 60)

        model = NeuralNetModel(
            name=name,
            lr=lr,
            **BASE_CONFIG,
        )

        t0 = time.time()
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=verbose)
        elapsed = time.time() - t0

        tr  = model.evaluate(X_train, y_train, split_name="train")
        val = model.evaluate(X_val,   y_val,   split_name="val")
        tst = model.evaluate(X_test,  y_test,  split_name="test")

        records.append({
            "lr":             lr,
            "name":           name,
            "epochs_trained": len(model.history.get("train_loss", [])),
            "train_time_s":   round(elapsed, 1),
            "Train_RMSE_log": round(tr["rmse_log"],  4),
            "Val_RMSE_log":   round(val["rmse_log"], 4),
            "Test_RMSE_log":  round(tst["rmse_log"], 4),
            "Val_R2":         round(val["r2_log"],   4),
            "Test_R2":        round(tst["r2_log"],   4),
            "Val_MAE_log":    round(val["mae_log"],  4),
            "Val_RMSE_raw":   round(val["rmse_raw"], 0),
            "Val_MAE_raw":    round(val["mae_raw"],  0),
            "Test_MAE_raw":   round(tst["mae_raw"],  0),
        })

        all_models[lr] = model

        logger.info(
            "  lr=%.0e  val_RMSE=%.4f  val_R²=%.4f  test_RMSE=%.4f  (%ds)",
            lr, val["rmse_log"], val["r2_log"], tst["rmse_log"], int(elapsed),
        )

    results_df = pd.DataFrame(records).set_index("lr").sort_values("Val_RMSE_log")

    best_lr    = float(results_df.index[0])
    best_model = all_models[best_lr]

    logger.info("\n%s", "=" * 60)
    logger.info("Best lr=%.0e  Val_RMSE=%.4f  Val_R²=%.4f",
                best_lr,
                results_df.loc[best_lr, "Val_RMSE_log"],
                results_df.loc[best_lr, "Val_R2"])

    return results_df, best_model, best_lr
