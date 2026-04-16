"""
main.py
────────
End-to-end pipeline runner for the Steam Sales Prediction project.

Usage
─────
    # Full pipeline (post-release model, baseline + advanced models)
    python main.py

    # Launch-time model only (no post-release features)
    python main.py --framing launch

    # Baselines only (faster for iteration)
    python main.py --models baseline

    # Save processed data and features to outputs/
    python main.py --save

Run  python main.py --help  for all options.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running from any directory
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from configs.config import (
    OUTPUT_DIR,
    POST_RELEASE_MODEL,
    PROCESSED_PATH,
    RAW_MERGED_PATH,
    TARGET_LOG,
    TARGET_RAW,
)
from src.data.loader import load_merged, validate_merged
from src.data.preprocessor import run_preprocessing_pipeline, audit_columns
from src.features.engineer import prepare_features
from src.models.baseline import run_all_baselines
from src.evaluation.metrics import compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Steam Sales Prediction — end-to-end pipeline"
    )
    parser.add_argument(
        "--framing",
        choices=["post_release", "launch"],
        default="post_release",
        help="Model framing: 'post_release' uses all observed features; "
             "'launch' uses only features available at launch time.",
    )
    parser.add_argument(
        "--models",
        choices=["baseline", "advanced", "all"],
        default="baseline",
        help="Which model tier to run.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save processed DataFrame and feature matrices to outputs/.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print a column audit table after preprocessing.",
    )
    return parser.parse_args()


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def step_load(args: argparse.Namespace) -> pd.DataFrame:
    logger.info("━━━ Step 1: Load data ━━━")
    df = load_merged(RAW_MERGED_PATH, verbose=True)
    validate_merged(df)
    return df


def step_preprocess(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    logger.info("━━━ Step 2: Preprocessing ━━━")
    post_release = (args.framing == "post_release")
    df = run_preprocessing_pipeline(df, post_release=post_release, verbose=True)

    if args.audit:
        audit = audit_columns(df)
        print("\nColumn audit (sorted by null %):\n")
        print(audit.to_string(index=False))

    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_PATH, index=False)
        logger.info("Saved preprocessed data → %s", PROCESSED_PATH)

    return df


def step_features(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    logger.info("━━━ Step 3: Feature engineering & splitting ━━━")
    post_release = (args.framing == "post_release")
    data = prepare_features(df, post_release=post_release, return_pipeline=True)

    logger.info(
        "Feature matrix shape: train=%s  val=%s  test=%s",
        data["X_train"].shape,
        data["X_val"].shape,
        data["X_test"].shape,
    )

    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        import numpy as np
        np.save(OUTPUT_DIR / "X_train.npy", data["X_train"])
        np.save(OUTPUT_DIR / "X_val.npy",   data["X_val"])
        np.save(OUTPUT_DIR / "X_test.npy",  data["X_test"])
        pd.Series(data["output_cols"]).to_csv(
            OUTPUT_DIR / "feature_names.csv", index=False, header=False
        )
        logger.info("Saved feature matrices → %s", OUTPUT_DIR)

    return data


def step_baseline_models(data: dict) -> pd.DataFrame:
    logger.info("━━━ Step 4a: Baseline models ━━━")
    summary = run_all_baselines(data)
    print("\nBaseline model comparison (validation set):\n")
    print(summary.to_string())
    return summary


def step_advanced_models(data: dict) -> pd.DataFrame:
    logger.info("━━━ Step 4b: Advanced models ━━━")
    from src.models.advanced import run_advanced_models
    summary = run_advanced_models(data, use_xgboost=True)
    print("\nAdvanced model comparison (validation set):\n")
    print(summary.to_string())
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logger.info("Framing: %s  |  Models: %s", args.framing, args.models)

    df   = step_load(args)
    df   = step_preprocess(df, args)
    data = step_features(df, args)

    all_summaries = {}

    if args.models in ("baseline", "all"):
        baseline_summary = step_baseline_models(data)
        all_summaries["baseline"] = baseline_summary

    if args.models in ("advanced", "all"):
        advanced_summary = step_advanced_models(data)
        all_summaries["advanced"] = advanced_summary

    if all_summaries:
        print("\n\n" + "═" * 60)
        print("  FINAL MODEL COMPARISON (validation RMSE_log, ascending)")
        print("═" * 60)
        combined = pd.concat(all_summaries.values())
        print(combined.sort_values("val_RMSE_log").to_string())

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
