"""
run_game_age_experiment.py
──────────────────────────
A/B experiment: does adding `game_age` (=2026 - release_year) help tree models
predict log_copies_sold, vs. the current baseline that uses `release_year` only?

Two arms are trained, each with all three tree models (RF, GB, XGBoost):

    arm = "year_only"   →  release_year only,  game_age  excluded   (BASELINE)
    arm = "age_only"    →  game_age    only,   release_year excluded (TREATMENT)

Outputs (none of these overwrite the original baseline files):
    Models/RandomForest_year_only.joblib       Models/RandomForest_age_only.joblib
    Models/GradientBoosting_year_only.joblib   Models/GradientBoosting_age_only.joblib
    Models/XGBoost_year_only.joblib            Models/XGBoost_age_only.joblib
    outputs/loss_curve_GradientBoosting_year_only.png   ...age_only.png
    outputs/loss_curve_XGBoost_year_only.png            ...age_only.png
    outputs/age_vs_year_summary.csv     (combined metrics for both arms)

Usage:
    python run_game_age_experiment.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from configs.config import MODELS_DIR, OUTPUT_DIR, POST_RELEASE_MODEL, PROCESSED_PATH
from src.features.engineer import prepare_features
from src.models.advanced import (
    AdvancedModel,
    GradientBoostModel,
    RandomForestModel,
    XGBoostModel,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ARMS = {
    "year_only": dict(use_release_year=True,  use_game_age=False),
    "age_only":  dict(use_release_year=False, use_game_age=True),
}


def save_loss_curve(model: AdvancedModel, suffix: str) -> Path | None:
    """Render and save the train/val loss curve for boosting models."""
    if model.loss_curve_ is None:
        logger.info("[%s] no loss curve — skipping figure.", model.name)
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    if "train" in model.loss_curve_:
        ax.plot(model.loss_curve_["train"], label="train RMSE")
    if "val" in model.loss_curve_:
        ax.plot(model.loss_curve_["val"], label="val RMSE")
    if "best_iteration" in model.loss_curve_:
        best = model.loss_curve_["best_iteration"]
        ax.axvline(best, color="red", linestyle="--",
                   label=f"best iter={best}")
    ax.set_xlabel("Iteration / tree")
    ax.set_ylabel("RMSE (log target)")
    ax.set_title(f"{model.name} — loss curve ({suffix})")
    ax.legend()
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUTPUT_DIR / f"loss_curve_{model.name}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("[%s] loss curve saved → %s", model.name, fig_path)
    return fig_path


def train_one_arm(arm_name: str, df: pd.DataFrame, toggles: dict) -> list[dict]:
    """Train all three models for one arm and return per-model metric rows."""
    logger.info("\n" + "=" * 70)
    logger.info("ARM: %s   toggles=%s", arm_name, toggles)
    logger.info("=" * 70)

    data = prepare_features(
        df,
        post_release=POST_RELEASE_MODEL,
        **toggles,
    )
    feature_cols = data["feature_cols"]
    logger.info("Features used (%d): release_year=%s  game_age=%s",
                len(feature_cols),
                "release_year" in feature_cols,
                "game_age" in feature_cols)

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    models: list[AdvancedModel] = [
        RandomForestModel(),
        GradientBoostModel(),
    ]
    try:
        models.append(XGBoostModel())
    except ImportError:
        logger.warning("XGBoost not installed — skipping that model.")

    # Suffix every model name so save() / loss-curve files don't collide
    # with the original baseline artefacts.
    for m in models:
        m.name = f"{m.name}_{arm_name}"

    rows = []
    for m in models:
        logger.info("[%s] training …", m.name)
        if isinstance(m, (GradientBoostModel, XGBoostModel)):
            m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            m.fit(X_train, y_train)

        tr  = m.evaluate(X_train, y_train, "train")
        val = m.evaluate(X_val,   y_val,   "val")
        tst = m.evaluate(X_test,  y_test,  "test")

        m.save()                       # → Models/<name>.joblib
        save_loss_curve(m, arm_name)   # no-op for RF (no curve)

        rows.append({
            "arm":            arm_name,
            "model":          m.name,
            "n_features":     len(feature_cols),
            "train_RMSE_log": tr["rmse_log"],
            "val_RMSE_log":   val["rmse_log"],
            "test_RMSE_log":  tst["rmse_log"],
            "val_MAE_log":    val["mae_log"],
            "val_R2_log":     val["r2_log"],
            "val_RMSE_raw":   val["rmse_raw"],
            "val_MAE_raw":    val["mae_raw"],
        })
        logger.info("[%s] val RMSE(log)=%.4f  val R²=%.4f",
                    m.name, val["rmse_log"], val["r2_log"])

    return rows


def main() -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found. Re-run notebooks/02_preprocessing.ipynb "
            "to regenerate the parquet with the new `game_age` column."
        )

    df = pd.read_parquet(PROCESSED_PATH)
    logger.info("Loaded processed data: %d rows × %d cols", *df.shape)

    if "game_age" not in df.columns:
        raise KeyError(
            "`game_age` column missing from processed.parquet. "
            "Re-run notebooks/02_preprocessing.ipynb after adding the new step."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for arm_name, toggles in ARMS.items():
        all_rows.extend(train_one_arm(arm_name, df, toggles))

    summary = pd.DataFrame(all_rows)
    summary_path = OUTPUT_DIR / "age_vs_year_summary.csv"
    summary.to_csv(summary_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY (lower RMSE / MAE = better, higher R² = better)")
    logger.info("=" * 70)
    print(summary.to_string(index=False))
    logger.info("\nSaved combined summary → %s", summary_path)


if __name__ == "__main__":
    main()
