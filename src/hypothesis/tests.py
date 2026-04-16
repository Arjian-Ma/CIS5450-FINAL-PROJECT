"""
src/hypothesis/tests.py
───────────────────────
Framework for simulation-based hypothesis testing.

Three tests are planned (per the project proposal + TA feedback):

  Test 1 – Free-to-Play vs Paid (exploratory comparison)
      H₀: Median copiesSold is the same for free and paid games.
      Hₐ: Free games have higher median sales.
      Method: permutation test on difference in medians (robust to skew).
      TA note: F2P estimates from Gamalytic are noisier; frame this as
               "exploratory comparison" rather than a central finding.

  Test 2 – Price Effect on Sales (regression coefficient test)
      H₀: The price coefficient in a linear model is ≥ 0 (no negative effect).
      Hₐ: Price is negatively associated with log_copies_sold.
      Method: permutation test on the OLS price coefficient.

  Test 3 – Publisher Class and Sales (multi-group comparison)
      H₀: Distribution of log_copies_sold is the same across publisher classes.
      Hₐ: At least one publisher class differs.
      Method: permutation-based ANOVA (F-statistic under null).

Status
──────
  All functions have complete signatures, docstrings, and structural stubs.
  The statistical implementation bodies are marked TODO — to be completed
  during the hypothesis-testing phase of the project.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class HypothesisResult:
    """Stores the outcome of a single hypothesis test."""
    test_name:      str
    observed_stat:  float
    p_value:        float
    n_permutations: int
    alpha:          float = 0.05
    extra:          dict  = None          # test-specific extras (CI, group stats …)

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    @property
    def reject_h0(self) -> bool:
        return self.p_value < self.alpha

    def summary(self) -> str:
        decision = "REJECT H₀" if self.reject_h0 else "FAIL TO REJECT H₀"
        return (
            f"\n{'─'*60}\n"
            f"  Test:           {self.test_name}\n"
            f"  Observed stat:  {self.observed_stat:.4f}\n"
            f"  p-value:        {self.p_value:.4f}\n"
            f"  α:              {self.alpha}\n"
            f"  Decision:       {decision} (n_perm={self.n_permutations:,})\n"
            f"{'─'*60}"
        )


# ── Test 1: Free-to-Play vs Paid ──────────────────────────────────────────────

def test_free_vs_paid(
    df: pd.DataFrame,
    target_col: str = "log_copies_sold",
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> HypothesisResult:
    """
    Permutation test: do free-to-play and paid games have the same
    median estimated sales?

    ⚠️  Gamalytic caveat: sales estimates for free games (especially small
    titles) carry higher uncertainty.  Interpret significant results as
    "associated with" rather than causal, and report the uncertainty.

    Parameters
    ----------
    df            : Preprocessed DataFrame containing `is_free_to_play` and
                    the target column.
    target_col    : Target to test on (use log scale to reduce outlier influence).
    n_permutations: Number of permutation iterations.
    alpha         : Significance level.

    Returns
    -------
    HypothesisResult

    Implementation notes (TODO)
    ───────────────────────────
    1. Split df into free_group and paid_group on is_free_to_play.
    2. Compute observed_stat = median(free) - median(paid).
    3. Pool the two groups, then repeatedly:
         a. Shuffle the pooled array.
         b. Re-split at the size of the free group.
         c. Compute the permuted statistic.
    4. p_value = proportion of permuted stats ≥ observed_stat (one-sided)
       or ≥ |observed_stat| (two-sided).
    5. Also compute bootstrap 95 % CI for the difference in medians.
    """
    # ── Guard clauses ──────────────────────────────────────────────────────
    _require_columns(df, ["is_free_to_play", target_col])

    free_group = df.loc[df["is_free_to_play"] == 1, target_col].dropna().values
    paid_group = df.loc[df["is_free_to_play"] == 0, target_col].dropna().values

    logger.info(
        "Test 1: free n=%d  paid n=%d", len(free_group), len(paid_group)
    )

    # ── TODO: implement permutation test ──────────────────────────────────
    observed_stat = float(np.median(free_group) - np.median(paid_group))
    p_value = _NOT_IMPLEMENTED_p_value()

    return HypothesisResult(
        test_name="Free-to-Play vs Paid (median copiesSold)",
        observed_stat=observed_stat,
        p_value=p_value,
        n_permutations=n_permutations,
        alpha=alpha,
        extra={
            "free_median_log":  float(np.median(free_group)),
            "paid_median_log":  float(np.median(paid_group)),
            "free_n":           len(free_group),
            "paid_n":           len(paid_group),
            # TODO: add bootstrap_ci key once implemented
        },
    )


# ── Test 2: Price Effect ───────────────────────────────────────────────────────

def test_price_effect(
    df: pd.DataFrame,
    price_col: str = "Price",          # already log-transformed in preprocessed df
    target_col: str = "log_copies_sold",
    control_cols: list[str] | None = None,
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> HypothesisResult:
    """
    Permutation test on the price coefficient in an OLS regression.

    H₀: β_price ≥ 0 (price has no negative effect on sales)
    Hₐ: β_price < 0 (higher price → fewer copies sold)

    We use a permutation test rather than a standard t-test because:
      - copiesSold is right-skewed even after log transform
      - the normality assumption for t-statistics may not hold at the tails

    Parameters
    ----------
    df            : Preprocessed DataFrame.
    price_col     : The (log-transformed) price column name.
    target_col    : Log-scale target.
    control_cols  : Additional columns to include as controls in the OLS model.
                    If None, defaults to a small set of plausible confounders.
    n_permutations: Number of permutation iterations.

    Implementation notes (TODO)
    ───────────────────────────
    1. Fit OLS(target ~ price + controls) on the full sample.
       Record observed β_price.
    2. Permutation loop: shuffle price_col, re-fit OLS, record permuted β.
    3. p_value = proportion of permuted β ≤ observed β  (one-sided, left tail).
    4. Also report the OLS 95 % CI for β_price for comparison.
    """
    _require_columns(df, [price_col, target_col])

    if control_cols is None:
        control_cols = [
            "publisher_class_ord",
            "reviewScore",
            "genre_indie",
            "genre_action",
            "earlyAccess",
        ]
    control_cols = [c for c in control_cols if c in df.columns]

    # ── TODO: implement permutation test ──────────────────────────────────
    observed_stat = _NOT_IMPLEMENTED_p_value()   # placeholder: OLS β_price
    p_value       = _NOT_IMPLEMENTED_p_value()

    return HypothesisResult(
        test_name="Price Effect on log_copies_sold (OLS permutation)",
        observed_stat=observed_stat,
        p_value=p_value,
        n_permutations=n_permutations,
        alpha=alpha,
        extra={
            "price_col":     price_col,
            "control_cols":  control_cols,
            # TODO: add ols_ci, ols_pvalue keys once implemented
        },
    )


# ── Test 3: Publisher Class ────────────────────────────────────────────────────

def test_publisher_class(
    df: pd.DataFrame,
    group_col: str = "publisher_class_ord",
    target_col: str = "log_copies_sold",
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> HypothesisResult:
    """
    Permutation-based ANOVA: is sales distribution the same across publisher
    classes (Hobbyist / Indie / AA / AAA)?

    H₀: All publisher classes have the same distribution of log_copies_sold.
    Hₐ: At least one publisher class differs.

    Test statistic: F-statistic (ratio of between-group variance to
    within-group variance).

    Implementation notes (TODO)
    ───────────────────────────
    1. Compute observed F on the real group labels.
    2. Permutation loop: shuffle group labels, recompute F.
    3. p_value = proportion of permuted F ≥ observed F.
    4. For a post-hoc analysis, compute pairwise bootstrap CIs between
       each pair of publisher classes.
    """
    _require_columns(df, [group_col, target_col])

    groups = df.groupby(group_col)[target_col].apply(list).to_dict()
    logger.info(
        "Test 3: publisher classes — %s",
        {k: len(v) for k, v in groups.items()},
    )

    # ── TODO: implement permutation F-test ────────────────────────────────
    observed_stat = _NOT_IMPLEMENTED_p_value()   # placeholder: F-statistic
    p_value       = _NOT_IMPLEMENTED_p_value()

    return HypothesisResult(
        test_name="Publisher Class Effect (permutation ANOVA)",
        observed_stat=observed_stat,
        p_value=p_value,
        n_permutations=n_permutations,
        alpha=alpha,
        extra={
            "group_sizes": {k: len(v) for k, v in groups.items()},
            "group_medians": {
                k: float(np.median(v)) for k, v in groups.items()
            },
            # TODO: add pairwise_ci key once implemented
        },
    )


# ── Utility: bootstrap confidence interval ────────────────────────────────────

def bootstrap_ci(
    group_a: np.ndarray,
    group_b: np.ndarray,
    stat_fn=np.median,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for stat_fn(group_a) - stat_fn(group_b).

    Returns (lower_bound, upper_bound) for the (1 - alpha) CI.

    TODO: implement the bootstrap loop.
    """
    raise NotImplementedError("bootstrap_ci: implementation pending.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")


def _NOT_IMPLEMENTED_p_value() -> float:
    """Placeholder that makes it obvious a test body hasn't been written yet."""
    return float("nan")


def run_all_tests(df: pd.DataFrame, n_permutations: int = 10_000) -> list[HypothesisResult]:
    """
    Run all three hypothesis tests and print a summary.

    Parameters
    ----------
    df             : Preprocessed DataFrame (output of preprocessor pipeline).
    n_permutations : Permutation iterations for each test.

    Returns
    -------
    List of HypothesisResult objects.
    """
    tests = [
        lambda: test_free_vs_paid(df, n_permutations=n_permutations),
        lambda: test_price_effect(df, n_permutations=n_permutations),
        lambda: test_publisher_class(df, n_permutations=n_permutations),
    ]

    results = []
    for fn in tests:
        try:
            result = fn()
            print(result.summary())
            results.append(result)
        except NotImplementedError as e:
            logger.warning("Test not yet implemented: %s", e)

    return results
