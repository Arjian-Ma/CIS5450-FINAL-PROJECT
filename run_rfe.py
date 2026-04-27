"""
run_rfe.py
──────────
CLI entry point for cross-validated recursive feature elimination.
Runs `select_features_rfecv` with sensible defaults and prints a readable summary.

Usage:
    python run_rfe.py                 # uses POST_RELEASE_MODEL from config
    python run_rfe.py --launch        # force launch-time framing (no leaky features)
    python run_rfe.py --step 5        # faster, coarser sweep
    python run_rfe.py --cv 5          # more stable, slower
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from configs.config import POST_RELEASE_MODEL, PROCESSED_PATH
from src.features.selection import select_features_rfecv

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--launch", action="store_true",
                   help="Use launch-time framing (drop post-release features).")
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--min-features", type=int, default=5)
    args = p.parse_args()

    post_release = (not args.launch) and POST_RELEASE_MODEL

    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols  (game_age in cols: {'game_age' in df.columns})")
    print(f"Mode: {'post-release' if post_release else 'launch-time'}   step={args.step}   cv={args.cv}\n")

    t0 = time.time()
    result = select_features_rfecv(
        df,
        post_release=post_release,
        step=args.step,
        cv=args.cv,
        min_features=args.min_features,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"RFECV finished in {elapsed/60:.1f} min")
    print(f"Best by min CV RMSE  : {result['n_optimal']} features  (RMSE log = {result['best_cv_rmse']:.4f})")
    print(f"Best by 1-SE rule    : {result['n_optimal_1se']} features  "
          "(smallest subset within 1 std-err of best — favours parsimony)")
    print("=" * 70)

    print("\nCV RMSE vs. number of features (every 4th subset):")
    cv = result["cv_table"]
    show = cv.iloc[::4].copy()
    if not show["chosen_by_min"].any():
        show = pd.concat([show, cv[cv["chosen_by_min"]]])
    if not show["chosen_by_1se"].any():
        show = pd.concat([show, cv[cv["chosen_by_1se"]]])
    show = show.sort_values("n_features").drop_duplicates("n_features")
    print(show[["n_features", "mean_cv_rmse_log", "std_cv_rmse_log",
                "chosen_by_min", "chosen_by_1se"]].to_string(index=False))

    print("\nTop 25 selected features (by importance in final estimator):")
    sel = result["ranking"].query("selected").head(25)
    print(sel[["feature", "importance_in_final"]].to_string(index=False))

    print("\nArtefacts in outputs/rfe/:")
    print("  rfecv_selected_ranking.csv   – feature-by-feature ranking + final importances")
    print("  rfecv_selected_cv_table.csv  – CV RMSE per subset size (use this to balance #features vs. accuracy)")
    print("  rfecv_selected_curve.png     – CV-RMSE-vs-#-features plot")


if __name__ == "__main__":
    main()
