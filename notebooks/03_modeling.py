# %% [markdown]
# # Notebook 3: Modelling
# **Steam Sales Prediction — CIS 5450 Final Project**
#
# Goals
# ─────
# 1. Fit and compare all baseline models (Mean, OLS, Ridge, Lasso)
# 2. Fit and compare advanced models (Random Forest, XGBoost)
# 3. Inspect Lasso feature selection and tree feature importances
# 4. Diagnose residuals and identify systematic errors
# 5. Pick the best model and report test-set performance

# %% [markdown]
# ## 0. Setup

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from configs.config import OUTPUT_DIR, TARGET_LOG
from src.features.engineer import prepare_features
from src.models.baseline import run_all_baselines, lasso_feature_importance, make_lasso
from src.models.advanced import run_advanced_models, RandomForestModel, XGBoostModel
from src.evaluation.metrics import compare_models, residual_df, quantile_rmse

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load preprocessed data & build feature matrix

# %%
df = pd.read_parquet("../outputs/processed.parquet")
print(f"Loaded: {df.shape}")

# Post-release framing (uses all observed features)
data = prepare_features(df, post_release=True, return_pipeline=True)
print(f"X_train: {data['X_train'].shape}  "
      f"X_val: {data['X_val'].shape}  "
      f"X_test: {data['X_test'].shape}")

# %% [markdown]
# ## 2. Baseline models

# %%
baseline_summary = run_all_baselines(data)
print("\nBaseline model comparison (validation set):")
baseline_summary

# %%
# Visualise baseline RMSE (log scale)
fig, ax = plt.subplots(figsize=(9, 4))
baseline_summary["val_RMSE_log"].sort_values(ascending=False).plot(kind="barh", ax=ax)
ax.set_xlabel("Validation RMSE (log scale)")
ax.set_title("Baseline model comparison")
plt.tight_layout()
plt.savefig("../outputs/model_baseline_rmse.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Lasso feature importance

# %%
lasso = make_lasso()
lasso.fit(data["X_train"], data["y_train"])

lasso_imp = lasso_feature_importance(lasso, data["output_cols"], top_n=30)
print(lasso_imp.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 9))
lasso_imp.set_index("feature")["coef"].sort_values().plot(kind="barh", ax=ax)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Lasso coefficients (top 30 non-zero features)")
plt.tight_layout()
plt.savefig("../outputs/model_lasso_coefs.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Advanced models

# %%
advanced_summary = run_advanced_models(data, use_xgboost=True)
print("\nAdvanced model comparison (validation set):")
advanced_summary

# %% [markdown]
# ## 5. Combined comparison

# %%
all_summary = pd.concat([baseline_summary, advanced_summary])
print("\nAll models — validation set:")
all_summary.sort_values("val_RMSE_log")

# %% [markdown]
# ## 6. Best model — feature importance

# %%
# Use the best-performing advanced model for feature importance
# (adapt model_name below after running the comparison above)
rf = RandomForestModel()
rf.fit(data["X_train"], data["y_train"])

rf_imp = rf.feature_importance_df(data["output_cols"], top_n=30)

fig, ax = plt.subplots(figsize=(9, 9))
rf_imp.set_index("feature")["importance"].sort_values().plot(kind="barh", ax=ax)
ax.set_title("Random Forest feature importances (top 30)")
plt.tight_layout()
plt.savefig("../outputs/model_rf_importance.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Residual diagnostics

# %%
# Use the best model on the validation set for residual analysis
y_pred_val = rf.predict(data["X_val"])
resid = residual_df(
    np.asarray(data["y_val"]),
    y_pred_val,
    metadata=data["X_val_raw"][["publisher_class_ord"]].reset_index(drop=True)
    if "publisher_class_ord" in data["X_val_raw"].columns else None,
)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Predicted vs actual
axes[0].scatter(resid["y_true_log"], resid["y_pred_log"], alpha=0.15, s=6)
lo, hi = resid["y_true_log"].min(), resid["y_true_log"].max()
axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1)
axes[0].set_xlabel("Actual log1p(copiesSold)")
axes[0].set_ylabel("Predicted log1p(copiesSold)")
axes[0].set_title("Predicted vs Actual")

# Residual distribution
resid["residual"].plot(kind="hist", bins=80, ax=axes[1])
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_xlabel("Residual (pred − actual)")
axes[1].set_title("Residual distribution")

# Residuals vs predicted
axes[2].scatter(resid["y_pred_log"], resid["residual"], alpha=0.15, s=6)
axes[2].axhline(0, color="red", linestyle="--")
axes[2].set_xlabel("Predicted value")
axes[2].set_ylabel("Residual")
axes[2].set_title("Residuals vs Predicted")

plt.tight_layout()
plt.savefig("../outputs/model_residuals.png", dpi=150)
plt.show()

# %%
# RMSE by sales quantile — are we better at predicting low or high sellers?
qrmse = quantile_rmse(np.asarray(data["y_val"]), y_pred_val)
print(qrmse.to_string(index=False))

# %% [markdown]
# ## 8. Test-set evaluation (run ONCE at the very end)
#
# ⚠️  Do NOT look at the test set repeatedly.  Run the cell below only when
# you have finalized your model choice.

# %%
# UNCOMMENT when model is finalised:
# best_model = rf   # or xgb_model, etc.
# y_pred_test = best_model.predict(data["X_test"])
# from src.evaluation.metrics import evaluate_predictions
# test_metrics = evaluate_predictions(
#     np.asarray(data["y_test"]), y_pred_test, model_name="BestModel_TEST"
# )
# print("Test set metrics:", test_metrics)

# %% [markdown]
# ## 9. Launch-time model (no post-release features)
#
# For fairness of the feature comparison, also run the best model in
# launch-time mode (only features observable at/before release).

# %%
data_launch = prepare_features(df, post_release=False, return_pipeline=True)
print(f"Launch-time feature matrix: {data_launch['X_train'].shape}")

# TODO: re-run best model with data_launch and compare val_RMSE_log
# This lets you quantify how much post-release signal is adding.
