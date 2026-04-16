# %% [markdown]
# # Notebook 2: Data Preprocessing
# **Steam Sales Prediction — CIS 5450 Final Project**
#
# Goals
# ─────
# Walk through each preprocessing step in `src/data/preprocessor.py`,
# inspect the result, and verify the final feature set is clean and ready
# for modelling.

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

from configs.config import RAW_MERGED_PATH, OUTPUT_DIR
from src.data.loader import load_merged, validate_merged
from src.data.preprocessor import (
    run_preprocessing_pipeline,
    audit_columns,
    PUBLISHER_CLASS_ORDER,
)
from src.features.engineer import prepare_features, get_feature_columns

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load raw data

# %%
df_raw = load_merged(RAW_MERGED_PATH, verbose=True)
validate_merged(df_raw)
print(f"Raw shape: {df_raw.shape}")

# %% [markdown]
# ## 2. Run the full preprocessing pipeline
#
# This calls all steps in order.  Set `verbose=True` to see each step logged.

# %%
df = run_preprocessing_pipeline(df_raw, post_release=True, verbose=True)
print(f"\nProcessed shape: {df.shape}")

# %% [markdown]
# ## 3. Column audit

# %%
audit = audit_columns(df)
print(audit.to_string(index=False))

# %% [markdown]
# ## 4. Verify parsed columns

# %%
# 4a. Release date
print("release_year  sample:", df["release_year"].dropna().value_counts().sort_index().tail(10))
print("release_month sample:", df["release_month"].value_counts().sort_index())

# %%
# 4b. Language count
print("language_count distribution:")
print(df["language_count"].value_counts().sort_index().head(20))

# %%
# 4c. Genre binary columns
genre_cols = [c for c in df.columns if c.startswith("genre_")]
print("Genre columns created:", genre_cols)
df[genre_cols].sum().sort_values(ascending=False)

# %%
# 4d. Category binary flags
cat_cols = [c for c in df.columns if c.startswith("has_")]
print("Category/feature flags:")
df[cat_cols].sum().sort_values(ascending=False)

# %%
# 4e. Tag columns
tag_cols = [c for c in df.columns if c.startswith("tag_")]
print(f"Number of tag binary columns: {len(tag_cols)}")
df[tag_cols].sum().sort_values(ascending=False).head(20)

# %%
# 4f. Publisher class encoding
print(df["publisher_class_ord"].value_counts().sort_index())
print("Mapping:", PUBLISHER_CLASS_ORDER)

# %%
# 4g. Derived features
derived = ["total_reviews", "review_ratio", "is_free_to_play",
           "platform_count", "description_length", "required_age_flag", "has_dlc"]
df[derived].describe()

# %% [markdown]
# ## 5. Target variable check

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

df["copiesSold"].plot(kind="hist", bins=100, logy=True, ax=axes[0])
axes[0].set_xlabel("copiesSold (raw)")
axes[0].set_title("Raw target distribution")

df["log_copies_sold"].plot(kind="hist", bins=80, ax=axes[1])
axes[1].set_xlabel("log1p(copiesSold)")
axes[1].set_title("Log-transformed target distribution")

plt.tight_layout()
plt.savefig("../outputs/prep_target_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Feature matrix overview

# %%
feature_cols = get_feature_columns(df, post_release=True)
print(f"Total feature columns (post-release): {len(feature_cols)}")

feature_cols_launch = get_feature_columns(df, post_release=False)
print(f"Total feature columns (launch-time): {len(feature_cols_launch)}")

# %%
# Null counts in feature matrix
X_check = df[feature_cols]
null_summary = X_check.isnull().sum()
print("Null counts per feature column (any > 0 shown):")
print(null_summary[null_summary > 0].sort_values(ascending=False))

# %% [markdown]
# ## 7. Feature / target correlation (post-release)

# %%
corr_with_target = (
    df[feature_cols + ["log_copies_sold"]]
    .select_dtypes(include="number")
    .corr()["log_copies_sold"]
    .drop("log_copies_sold")
    .sort_values(key=abs, ascending=False)
)

print("Top 20 features correlated with log_copies_sold:")
print(corr_with_target.head(20).to_string())

# %%
# Visual
fig, ax = plt.subplots(figsize=(8, 10))
corr_with_target.head(30).sort_values().plot(kind="barh", ax=ax)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top-30 feature correlations with log_copies_sold")
plt.tight_layout()
plt.savefig("../outputs/prep_feature_correlation.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Save processed data

# %%
df.to_parquet("../outputs/processed.parquet", index=False)
print("Saved → outputs/processed.parquet")

# %% [markdown]
# ## Summary of preprocessing decisions
#
# | Issue | Decision |
# |---|---|
# | `copiesSold` extreme right-skew | Use `log1p(copiesSold)` as target |
# | `Movies`, `Score rank`, `Reviews`, `Notes` near-entirely null | Dropped |
# | `URL` columns (Header image, steamUrl, Screenshots) | Dropped |
# | `User score` 99.97 % zero | Dropped |
# | `Metacritic score` 96 % zero | Kept; added `has_metacritic` binary flag |
# | `Supported languages` list string | Parsed → `language_count` |
# | `Genres` / `Categories` / `Tags` comma-separated strings | Parsed → binary columns |
# | `Estimated owners` range string | Parsed → midpoint; flagged as post-release |
# | `Release date` (two overlapping columns) | Both parsed → `release_year/month/quarter` |
# | `Developers`, `Publishers` high cardinality | Frequency-encoded |
# | `publisherClass` ordinal | Encoded 0-3 |
# | Post-release features (playtime, review counts) | Conditionally excluded via `post_release` flag |
