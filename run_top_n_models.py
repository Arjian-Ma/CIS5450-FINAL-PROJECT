"""
run_top_n_models.py
───────────────────
Train RandomForest, GradientBoosting, and XGBoost on the top-N features chosen
by recursive feature elimination.  Use this to test whether a leaner feature
set holds up against the full-feature baseline.

Filenames are suffixed with `_top{N}_{framing}` so nothing collides with the
existing baseline artefacts in Models/ or the experiment outputs.

Usage:
    python run_top_n_models.py                  # default: top-60 features, post-release framing
    python run_top_n_models.py --n-features 40  # top-40
    python run_top_n_models.py --launch         # drop post-release leaky features
    python run_top_n_models.py --n-features 30 --launch
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from configs.config import (
    OUTPUT_DIR,
    POST_RELEASE_MODEL,
    PROCESSED_PATH,
    TARGET_LOG,
    TARGET_RAW,
)
from src.features.engineer import prepare_features
from src.features.selection import (
    load_top_n_from_ranking,
    select_features_rfe,
    transform_to_selected,
)
from src.models.advanced import (
    AdvancedModel,
    GradientBoostModel,
    RandomForestModel,
    XGBoostModel,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT_DIR: Path = OUTPUT_DIR / "rfe" / "models"
DEFAULT_RANKING_PATH: Path = OUTPUT_DIR / "rfe" / "rfecv_selected_ranking.csv"


def save_loss_curve(model: AdvancedModel) -> None:
    """Render and save train/val loss curve for boosting models. No-op for RF."""
    if model.loss_curve_ is None:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    if "train" in model.loss_curve_:
        ax.plot(model.loss_curve_["train"], label="train RMSE")
    if "val" in model.loss_curve_:
        ax.plot(model.loss_curve_["val"], label="val RMSE")
    if "best_iteration" in model.loss_curve_:
        b = model.loss_curve_["best_iteration"]
        ax.axvline(b, color="red", linestyle="--", label=f"best iter={b}")
    ax.set_xlabel("Iteration / tree")
    ax.set_ylabel(
        "RMSE (raw copiesSold)" if model.target_scale == "raw"
        else "RMSE (log target)"
    )
    ax.set_title(f"{model.name} — loss curve")
    ax.legend()
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"loss_curve_{model.name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("[%s] loss curve saved → %s", model.name, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-features", type=int, default=60,
                   help="Number of top features to keep (default: 60).")
    p.add_argument("--launch", action="store_true",
                   help="Use launch-time framing (drop post-release features).")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving models, loss curves, and summary CSV.")
    p.add_argument("--ranking-path", type=str, default=str(DEFAULT_RANKING_PATH),
                   help=f"Cached RFE ranking CSV to read top-N from "
                        f"(default: {DEFAULT_RANKING_PATH}).")
    p.add_argument("--refit-rfe", action="store_true",
                   help="Force RFE to refit instead of reading the cached ranking.")
    p.add_argument("--target", choices=["log", "raw"], default="log",
                   help="Target scale: 'log' = log1p(copiesSold) (default, "
                        "well-behaved for skewed sales); 'raw' = copiesSold "
                        "directly (interpretable but dominated by blockbusters).")
    p.add_argument("--xgb-objective",
                   choices=["squarederror", "tweedie", "gamma",
                            "absoluteerror", "huber", "quantile"],
                   default="squarederror",
                   help="XGBoost loss. squarederror=default mean-RMSE; "
                        "tweedie/gamma=distribution-based skew correction "
                        "(force --target raw); absoluteerror/huber/quantile "
                        "are robust losses that work on either target scale.")
    p.add_argument("--tweedie-power", type=float, default=1.5,
                   help="tweedie_variance_power for Tweedie objective. "
                        "1.0=Poisson, 2.0=Gamma, 1.5 (default) is the typical "
                        "compound Poisson-Gamma sweet spot. Must be in [1.0, 2.0).")
    p.add_argument("--quantile-alpha", type=float, default=0.5,
                   help="Target quantile for quantile regression "
                        "(0.5 = median, 0.9 = upper tail). Default 0.5.")
    p.add_argument("--huber-slope", type=float, default=1.0,
                   help="huber_slope (delta) for pseudo-Huber loss. Smaller "
                        "= more robust to outliers, but slower convergence.")
    args = p.parse_args()

    # Distribution-based objectives need raw positive target; auto-force it.
    if args.xgb_objective in {"tweedie", "gamma"}:
        if args.target != "raw":
            print(f"⚠  --xgb-objective {args.xgb_objective} requires raw target. "
                  "Forcing --target raw.")
            args.target = "raw"

    post_release = (not args.launch) and POST_RELEASE_MODEL
    framing = "launch" if not post_release else "postrelease"
    target_col = TARGET_RAW if args.target == "raw" else TARGET_LOG

    obj_tag = "" if args.xgb_objective == "squarederror" else f"_{args.xgb_objective}"
    if args.xgb_objective == "tweedie":
        obj_tag += f"{args.tweedie_power:g}"           # e.g. _tweedie1.5
    elif args.xgb_objective == "quantile":
        obj_tag += f"{args.quantile_alpha:g}"          # e.g. _quantile0.5
    elif args.xgb_objective == "huber":
        obj_tag += f"{args.huber_slope:g}"             # e.g. _huber1
    suffix = f"top{args.n_features}_{framing}_{args.target}{obj_tag}"

    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Mode: {framing}   Top-N: {args.n_features}   "
          f"Target: {args.target} ({target_col})   "
          f"XGB objective: {args.xgb_objective}"
          + (f"  (power={args.tweedie_power})" if args.xgb_objective == "tweedie" else "")
          + "\n")

    # ── Step 1: pick top-N features (cached ranking by default) ──────────────
    ranking_path = Path(args.ranking_path)
    use_cache = ranking_path.exists() and not args.refit_rfe

    if use_cache:
        print(f"→ Reading top {args.n_features} features from cached ranking: {ranking_path}")
        selected_features = load_top_n_from_ranking(ranking_path, args.n_features)
        data = prepare_features(
            df,
            target_col=target_col,
            post_release=post_release,
            use_release_year=False,
            use_game_age=True,
        )
    else:
        if args.refit_rfe:
            print("→ --refit-rfe set; running RFE from scratch.")
        else:
            print(f"→ No cached ranking at {ranking_path}; running RFE from scratch.")
        t0 = time.time()
        # RFE itself always uses the log target (more stable importance ranking
        # — raw target lets blockbusters dominate the importances).
        sel = select_features_rfe(
            df,
            n_features_to_select=args.n_features,
            post_release=post_release,
            save=not args.no_save,
            save_stem=f"rfe_{framing}",
        )
        print(f"   RFE done in {(time.time() - t0)/60:.1f} min")
        selected_features = sel["selected_features"]
        # If user asked for raw target, rebuild the data dict on the raw scale
        # (RFE's data dict is on log scale).
        if args.target == "raw":
            data = prepare_features(
                df,
                target_col=target_col,
                post_release=post_release,
                use_release_year=False,
                use_game_age=True,
            )
        else:
            data = sel["data"]

    print(f"   {len(selected_features)} features kept\n")

    # ── Step 2: reduce the train/val/test matrices to selected columns ───────
    reduced = transform_to_selected(data, selected_features)
    Xt, Xv, Xte = reduced["X_train"], reduced["X_val"], reduced["X_test"]
    yt, yv, yte = reduced["y_train"], reduced["y_val"], reduced["y_test"]

    # ── Step 3: train tree models on the reduced matrix ──────────────────────
    # Build the XGBoost params for the requested objective.
    xgb_extra: dict = {}
    if args.xgb_objective == "tweedie":
        xgb_extra = {
            "objective": "reg:tweedie",
            "tweedie_variance_power": args.tweedie_power,
        }
    elif args.xgb_objective == "gamma":
        xgb_extra = {"objective": "reg:gamma"}
    elif args.xgb_objective == "absoluteerror":
        xgb_extra = {"objective": "reg:absoluteerror"}
    elif args.xgb_objective == "huber":
        xgb_extra = {
            "objective": "reg:pseudohubererror",
            "huber_slope": args.huber_slope,
        }
    elif args.xgb_objective == "quantile":
        xgb_extra = {
            "objective": "reg:quantileerror",
            "quantile_alpha": args.quantile_alpha,
        }

    if args.xgb_objective != "squarederror":
        # Skew-fix experiment: only XGBoost can swap loss; RF/GB always use
        # squared error so they'd just duplicate previous baseline runs.
        models: list[AdvancedModel] = []
        try:
            models.append(XGBoostModel(**xgb_extra))
        except ImportError:
            logger.warning("xgboost not installed — nothing to do for skew-fix run.")
            return
    else:
        models = [RandomForestModel(), GradientBoostModel()]
        try:
            models.append(XGBoostModel())
        except ImportError:
            logger.warning("xgboost not installed — skipping.")

    # Suffix names so save() / loss curves don't overwrite earlier artefacts.
    # Tag each model with the target scale so evaluate() reports both scales correctly.
    for m in models:
        m.name = f"{m.name}_{suffix}"
        m.target_scale = args.target

    rows = []
    for m in models:
        print(f"[{m.name}] training …")
        if isinstance(m, (GradientBoostModel, XGBoostModel)):
            m.fit(Xt, yt, X_val=Xv, y_val=yv)
        else:
            m.fit(Xt, yt)

        tr  = m.evaluate(Xt,  yt,  "train")
        val = m.evaluate(Xv,  yv,  "val")
        tst = m.evaluate(Xte, yte, "test")

        if not args.no_save:
            m.save()              # → Models/<name>.joblib
            save_loss_curve(m)    # → outputs/rfe/models/loss_curve_<name>.png

        rows.append({
            "model":          m.name,
            "framing":        framing,
            "target_trained": args.target,
            "xgb_objective":  args.xgb_objective,
            "tweedie_power":  args.tweedie_power if args.xgb_objective == "tweedie" else "",
            "n_features":     args.n_features,
            "train_RMSE_log": tr["rmse_log"],
            "val_RMSE_log":   val["rmse_log"],
            "test_RMSE_log":  tst["rmse_log"],
            "val_MAE_log":    val["mae_log"],
            "val_R2_log":     val["r2_log"],
            "val_RMSE_raw":   val["rmse_raw"],
            "val_MAE_raw":    val["mae_raw"],
        })
        print(f"[{m.name}] val RMSE(log)={val['rmse_log']:.4f}  R²={val['r2_log']:.4f}\n")

    summary = pd.DataFrame(rows)
    if not args.no_save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = OUT_DIR / f"summary_{suffix}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary → {summary_path}")

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
