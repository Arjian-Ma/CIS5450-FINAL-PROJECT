"""
src/data/loader.py
──────────────────
Responsible only for reading raw CSV files off disk and running basic
sanity-checks.  No feature engineering happens here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_merged(path: Path | str, *, verbose: bool = True) -> pd.DataFrame:
    """
    Load the pre-merged dataset (games_merged.csv) produced by Data/merge_games.py.

    Parameters
    ----------
    path    : Path to games_merged.csv
    verbose : If True, print a short summary after loading.

    Returns
    -------
    Raw DataFrame — no cleaning applied yet.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Merged CSV not found at {path}.\n"
            "Run Data/merge_games.py first to produce games_merged.csv."
        )

    logger.info("Loading %s …", path)
    df = pd.read_csv(path, low_memory=False)

    if verbose:
        _print_summary(df, label="games_merged.csv")

    return df


def load_games(path: Path | str, *, verbose: bool = False) -> pd.DataFrame:
    """Load the raw Kaggle games.csv (Steam listing metadata)."""
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)
    if verbose:
        _print_summary(df, label="games.csv")
    return df


def load_gamelist(path: Path | str, *, verbose: bool = False) -> pd.DataFrame:
    """Load the raw Gamalytic games-list.csv (sales estimates)."""
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)
    if verbose:
        _print_summary(df, label="games-list.csv")
    return df


def validate_merged(df: pd.DataFrame) -> None:
    """
    Run lightweight assertions to catch obvious data-quality issues early.
    Raises ValueError if a critical check fails.
    """
    required_cols = {"AppID", "Name", "copiesSold", "Price", "publisherClass",
                     "Genres", "Categories", "Tags", "Positive", "Negative",
                     "earlyAccess", "firstReleaseDate", "reviewScore"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Merged DataFrame is missing expected columns: {missing}")

    null_target = df["copiesSold"].isna().sum()
    if null_target > 50:          # 9 expected; flag if suspiciously many
        logger.warning("copiesSold has %d null values — check the merge.", null_target)

    if df["AppID"].duplicated().any():
        n_dups = df["AppID"].duplicated().sum()
        logger.warning("AppID has %d duplicate values.", n_dups)

    logger.info("Validation passed: %d rows × %d cols", *df.shape)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")
    null_pct = df.isnull().mean().mul(100)
    high_null = null_pct[null_pct > 10].sort_values(ascending=False)
    if not high_null.empty:
        print(f"\n  Columns with >10% nulls:")
        for col, pct in high_null.items():
            print(f"    {col:<35} {pct:.1f}%")
    print(f"{'='*60}\n")
