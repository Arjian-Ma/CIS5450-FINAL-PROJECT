# %% [markdown]
# # Notebook 4: Hypothesis Testing
# **Steam Sales Prediction — CIS 5450 Final Project**
#
# Goals
# ─────
# Conduct three simulation-based hypothesis tests using the preprocessed data.
# All tests use permutation methods (not parametric t/F tests) because
# `copiesSold` is right-skewed and standard normality assumptions do not hold.
#
# Tests
# ─────
# | # | Question | Statistic | Method |
# |---|----------|-----------|--------|
# | 1 | Do free-to-play games sell more copies than paid games? | Difference in medians | Permutation test |
# | 2 | Is higher price associated with fewer copies sold? | OLS price coefficient | Permutation test |
# | 3 | Does publisher class affect the sales distribution? | F-statistic (ANOVA) | Permutation ANOVA |
#
# **TA guidance**: Frame Test 1 as an exploratory comparison rather than a
# central finding — Gamalytic estimates for small free games carry more noise.

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

from src.hypothesis.tests import (
    test_free_vs_paid,
    test_price_effect,
    test_publisher_class,
    run_all_tests,
    bootstrap_ci,
)

sns.set_theme(style="whitegrid")
N_PERM = 10_000     # increase to 50_000 for final report

# %% [markdown]
# ## 1. Load preprocessed data

# %%
df = pd.read_parquet("../outputs/processed.parquet")
print(f"Loaded: {df.shape}")

# Quick sanity check
required = ["log_copies_sold", "is_free_to_play", "Price",
            "publisher_class_ord", "reviewScore"]
assert all(c in df.columns for c in required), "Missing expected columns!"
print("All required columns present.")

# %% [markdown]
# ## 2. Exploratory: Free-to-Play vs Paid sales distributions

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, (label, mask) in zip(axes, [
    ("Free-to-Play (Price = 0)", df["is_free_to_play"] == 1),
    ("Paid (Price > 0)",         df["is_free_to_play"] == 0),
]):
    df.loc[mask, "log_copies_sold"].plot(kind="hist", bins=60, ax=ax, density=True)
    ax.set_title(f"{label}\n(n = {mask.sum():,})")
    ax.set_xlabel("log1p(copiesSold)")

plt.tight_layout()
plt.savefig("../outputs/hyp1_ftp_vs_paid_dist.png", dpi=150)
plt.show()

# %%
# Group summary statistics
for label, mask in [("Free", df["is_free_to_play"] == 1),
                     ("Paid", df["is_free_to_play"] == 0)]:
    subset = df.loc[mask, "log_copies_sold"]
    print(f"{label:5s}  n={len(subset):6,d}  "
          f"median={subset.median():.3f}  mean={subset.mean():.3f}  "
          f"std={subset.std():.3f}")

# %% [markdown]
# ## 3. Test 1: Free-to-Play vs Paid (permutation test on median difference)

# %%
result1 = test_free_vs_paid(df, n_permutations=N_PERM)
print(result1.summary())

# %%
# TODO (once test_free_vs_paid is implemented): plot the permutation distribution
#
# fig, ax = plt.subplots(figsize=(9, 4))
# ax.hist(result1.extra["null_distribution"], bins=100, density=True, label="Null distribution")
# ax.axvline(result1.observed_stat, color="red", label=f"Observed stat = {result1.observed_stat:.3f}")
# ax.set_xlabel("Permuted difference in medians")
# ax.set_title("Test 1: Permutation distribution (free − paid median log-sales)")
# ax.legend()
# plt.tight_layout()
# plt.savefig("../outputs/hyp1_perm_dist.png", dpi=150)
# plt.show()

# %% [markdown]
# ## 4. Exploratory: Price vs log sales (for paid games only)

# %%
paid = df[df["is_free_to_play"] == 0].copy()

# Price in the processed df is log1p-transformed — convert back for display
paid["price_display"] = np.expm1(paid["Price"])

fig, ax = plt.subplots(figsize=(9, 5))
sample = paid.sample(min(5000, len(paid)), random_state=42)
ax.scatter(sample["price_display"], sample["log_copies_sold"],
           alpha=0.2, s=8)
ax.set_xlabel("Price (USD, original scale)")
ax.set_ylabel("log1p(copiesSold)")
ax.set_title("Price vs log sales (paid games, random sample of 5 000)")
plt.tight_layout()
plt.savefig("../outputs/hyp2_price_scatter.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Test 2: Price effect (OLS permutation test)

# %%
result2 = test_price_effect(df, n_permutations=N_PERM)
print(result2.summary())

# %% [markdown]
# ## 6. Exploratory: Sales by publisher class

# %%
pub_labels = {0: "Hobbyist", 1: "Indie", 2: "AA", 3: "AAA"}

fig, ax = plt.subplots(figsize=(10, 5))
df.groupby("publisher_class_ord")["log_copies_sold"].apply(list)
data_groups = [
    df.loc[df["publisher_class_ord"] == k, "log_copies_sold"].dropna().values
    for k in sorted(pub_labels)
]
ax.boxplot(data_groups,
           labels=[pub_labels[k] for k in sorted(pub_labels)],
           showfliers=False)
ax.set_ylabel("log1p(copiesSold)")
ax.set_title("Sales distribution by publisher class (outliers hidden)")
plt.tight_layout()
plt.savefig("../outputs/hyp3_publisher_boxplot.png", dpi=150)
plt.show()

# %%
# Group medians
for k, label in sorted(pub_labels.items()):
    subset = df.loc[df["publisher_class_ord"] == k, "log_copies_sold"]
    print(f"{label:10s}  n={len(subset):6,d}  median={subset.median():.3f}")

# %% [markdown]
# ## 7. Test 3: Publisher class effect (permutation ANOVA)

# %%
result3 = test_publisher_class(df, n_permutations=N_PERM)
print(result3.summary())

# %% [markdown]
# ## 8. Bootstrap confidence intervals (planned)
#
# Once `bootstrap_ci` is implemented in `src/hypothesis/tests.py`:

# %%
# TODO: compute bootstrap CIs for Test 1 and Test 3 pairwise differences
#
# free_log  = df.loc[df["is_free_to_play"] == 1, "log_copies_sold"].values
# paid_log  = df.loc[df["is_free_to_play"] == 0, "log_copies_sold"].values
# ci_low, ci_high = bootstrap_ci(free_log, paid_log, stat_fn=np.median, n_bootstrap=10_000)
# print(f"95% CI for (free median − paid median): ({ci_low:.3f}, {ci_high:.3f})")

# %% [markdown]
# ## 9. Summary table

# %%
summary_table = pd.DataFrame([
    {
        "Test":             "1. Free vs Paid",
        "H₀":              "median sales equal",
        "Statistic":       "Δ median",
        "Observed":        result1.observed_stat,
        "p-value":         result1.p_value,
        "Decision":        "REJECT H₀" if result1.reject_h0 else "Fail to reject",
    },
    {
        "Test":             "2. Price Effect",
        "H₀":              "β_price ≥ 0",
        "Statistic":       "OLS β_price",
        "Observed":        result2.observed_stat,
        "p-value":         result2.p_value,
        "Decision":        "REJECT H₀" if result2.reject_h0 else "Fail to reject",
    },
    {
        "Test":             "3. Publisher Class",
        "H₀":              "all classes equal",
        "Statistic":       "F-statistic",
        "Observed":        result3.observed_stat,
        "p-value":         result3.p_value,
        "Decision":        "REJECT H₀" if result3.reject_h0 else "Fail to reject",
    },
])
print(summary_table.to_string(index=False))

# %% [markdown]
# ## 10. Caveats and write-up notes
#
# - **Gamalytic noise**: `copiesSold` is an *estimate*, not Steam-reported data.
#   For small free games the estimation error is larger.  State this in the
#   report and avoid causal language.
#
# - **Confounding**: Publisher class, price, and F2P status are correlated with
#   each other.  A significant result does not imply a single causal mechanism.
#
# - **Permutation tests**: We use permutation rather than parametric tests
#   because the log-transformed target is still mildly non-normal at the tails.
#   Permutation tests are exact under the exchangeability assumption.
#
# - **Multiple testing**: Running three tests at α = 0.05 inflates the
#   family-wise error rate.  Consider a Bonferroni correction (α* = 0.05 / 3 ≈ 0.017)
#   or simply report all three p-values and let the reader decide.
