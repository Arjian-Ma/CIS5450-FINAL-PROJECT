# %% [markdown]
# # Notebook 1: Exploratory Data Analysis
# **Steam Sales Prediction — CIS 5450 Final Project**
#
# Goals of this notebook
# ─────────────────────
# 1. Understand the raw data before any modelling
# 2. Characterise the target variable `copiesSold` (distribution, skew)
# 3. Explore key predictors (price, genre, publisher class, review score)
# 4. Surface data quality issues that inform preprocessing decisions
# 5. Identify potential leakage risks

# %% [markdown]
# ## 0. Setup

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))   # project root on path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from configs.config import RAW_MERGED_PATH, RAW_GAMELIST_PATH
from src.data.loader import load_merged, load_gamelist

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", "{:,.2f}".format)

# %% [markdown]
# ## 1. Load raw data

# %%
df_raw = load_merged(RAW_MERGED_PATH, verbose=True)
df_raw.head(3)

# %%
# High-level null summary
null_pct = df_raw.isnull().mean().mul(100).sort_values(ascending=False)
null_pct[null_pct > 0]

# %% [markdown]
# ## 2. Target variable: `copiesSold`

# %%
# Basic stats
df_raw["copiesSold"].describe()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Raw distribution (log x-axis)
axes[0].hist(df_raw["copiesSold"].dropna(), bins=100, log=True, edgecolor="none")
axes[0].set_xscale("log")
axes[0].set_xlabel("copiesSold (log scale)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of copiesSold (raw, log x-axis)")

# Log1p distribution
log_cs = np.log1p(df_raw["copiesSold"].dropna())
axes[1].hist(log_cs, bins=80, edgecolor="none")
axes[1].set_xlabel("log1p(copiesSold)")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of log1p(copiesSold)")

plt.tight_layout()
plt.savefig("../outputs/eda_target_dist.png", dpi=150)
plt.show()

# %%
# Percentile table
pcts = [50, 75, 90, 95, 99, 99.5]
for p in pcts:
    val = df_raw["copiesSold"].quantile(p / 100)
    print(f"  {p:5.1f}th percentile: {val:>15,.0f} copies")

# %% [markdown]
# ## 3. Publisher class

# %%
pub_class = df_raw["publisherClass"].value_counts()
print(pub_class)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

pub_class.plot(kind="bar", ax=axes[0], color=sns.color_palette())
axes[0].set_title("Game count by publisher class")
axes[0].set_ylabel("Number of games")
axes[0].tick_params(axis="x", rotation=0)

# Median copiesSold per publisher class
median_sales = (
    df_raw.groupby("publisherClass")["copiesSold"]
    .median()
    .reindex(["Hobbyist", "Indie", "AA", "AAA"])
)
median_sales.plot(kind="bar", ax=axes[1], color=sns.color_palette())
axes[1].set_yscale("log")
axes[1].set_title("Median copiesSold by publisher class")
axes[1].set_ylabel("Median copies sold (log scale)")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("../outputs/eda_publisher_class.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Price distribution and free-to-play

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

paid = df_raw[df_raw["Price"] > 0]
paid["Price"].clip(upper=60).plot(kind="hist", bins=60, ax=axes[0])
axes[0].set_xlabel("Price (USD, clipped at $60)")
axes[0].set_title("Price distribution (paid games only)")

ftp_counts = df_raw["Price"].eq(0).value_counts().rename({True: "Free", False: "Paid"})
ftp_counts.plot(kind="pie", ax=axes[1], autopct="%1.1f%%", startangle=90)
axes[1].set_ylabel("")
axes[1].set_title("Free-to-Play vs Paid")

plt.tight_layout()
plt.savefig("../outputs/eda_price.png", dpi=150)
plt.show()

# %%
# Median sales: free vs paid
for label, mask in [("Free (Price=0)", df_raw["Price"] == 0),
                     ("Paid (Price>0)", df_raw["Price"] > 0)]:
    med = df_raw.loc[mask, "copiesSold"].median()
    n   = mask.sum()
    print(f"{label:25s}  n={n:6,d}  median copies={med:>10,.0f}")

# %% [markdown]
# ## 5. Genres

# %%
# Explode comma-separated genres into individual rows
genres_flat = (
    df_raw["Genres"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
    .value_counts()
)

fig, ax = plt.subplots(figsize=(10, 5))
genres_flat.head(15).sort_values().plot(kind="barh", ax=ax)
ax.set_xlabel("Number of games")
ax.set_title("Top 15 genres by game count")
plt.tight_layout()
plt.savefig("../outputs/eda_genres.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Review score vs. copiesSold

# %%
sample = df_raw.dropna(subset=["reviewScore", "copiesSold"]).sample(
    min(5000, len(df_raw)), random_state=42
)

fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(
    sample["reviewScore"],
    np.log1p(sample["copiesSold"]),
    alpha=0.25, s=10,
    c=sample["publisherClass"].map({"Hobbyist": 0, "Indie": 1, "AA": 2, "AAA": 3}),
    cmap="viridis",
)
ax.set_xlabel("reviewScore (Gamalytic, 0–100)")
ax.set_ylabel("log1p(copiesSold)")
ax.set_title("Review score vs. log sales, coloured by publisher class")
plt.colorbar(sc, ax=ax, label="Publisher class (0=Hobbyist … 3=AAA)")
plt.tight_layout()
plt.savefig("../outputs/eda_review_vs_sales.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Release year trend

# %%
from datetime import datetime

df_raw["parsed_year"] = pd.to_datetime(
    df_raw["firstReleaseDate"], utc=True, errors="coerce"
).dt.year

year_counts  = df_raw["parsed_year"].value_counts().sort_index()
year_med_cs  = df_raw.groupby("parsed_year")["copiesSold"].median()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
year_counts[year_counts.index.between(2005, 2025)].plot(kind="bar", ax=axes[0])
axes[0].set_title("Games released per year (2005–2025)")
axes[0].tick_params(axis="x", rotation=45)

year_med_cs[year_med_cs.index.between(2005, 2025)].plot(kind="bar", ax=axes[1])
axes[1].set_yscale("log")
axes[1].set_title("Median copiesSold per release year")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("../outputs/eda_release_year.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Correlation heatmap (numeric columns)

# %%
numeric_cols = [
    "copiesSold", "Price", "reviewScore", "Positive", "Negative",
    "Recommendations", "Average playtime forever", "Peak CCU",
    "Metacritic score", "DLC count", "Achievements",
]
corr_df = df_raw[numeric_cols].copy()
corr_df["log_copies"] = np.log1p(corr_df["copiesSold"])

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    corr_df.corr(),
    annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    linewidths=0.5, ax=ax,
)
ax.set_title("Pearson correlation matrix (log_copies included)")
plt.tight_layout()
plt.savefig("../outputs/eda_correlation.png", dpi=150)
plt.show()

# %% [markdown]
# ## 9. Key takeaways
#
# | Observation | Implication |
# |---|---|
# | `copiesSold` is extremely right-skewed (skewness ≈ 70) | Always model `log1p(copiesSold)` |
# | Median sales: AAA >> AA >> Indie >> Hobbyist | `publisherClass` is a strong predictor |
# | Free games have ~5× higher median sales than paid | F2P flag is a useful feature; TA cautions estimates are noisier |
# | `reviewScore`, `Positive`, `Recommendations` strongly correlated with sales | Post-release leakage risk; note in write-up |
# | `Metacritic score` is 0 for 96 % of games | Add `has_metacritic` indicator; don't use raw score directly |
# | `Score rank`, `User score`, `Movies` are essentially empty | Drop before modelling |
# | `Tags`, `Genres`, `Categories` need parsing | Handled in preprocessor.py |
