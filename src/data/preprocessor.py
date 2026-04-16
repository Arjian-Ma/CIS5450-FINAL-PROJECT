"""
src/data/preprocessor.py
────────────────────────
All messy cleaning and structural transformations on the raw merged DataFrame.
Each public function is pure (returns a new/copy DataFrame) and has a single
responsibility.  Call `run_preprocessing_pipeline()` to run the full sequence.

What lives here
───────────────
  - Column drops (useless / near-all-null / URL columns)
  - Null handling for the target and key features
  - Date parsing → release_year, release_month
  - Structured-string parsing:
      Supported languages  → language_count
      Estimated owners     → estimated_owners_midpoint  (leakage flag noted)
      Genres               → genre_* binary columns
      Categories           → category_* binary flags
      Tags                 → tag_* binary columns (top-N) + tag_count
  - Derived numeric features (review counts, is_free_to_play, platform_count …)
  - publisherClass ordinal encoding
  - Log1p transform of heavily skewed numeric columns
  - Developer / Publisher frequency encoding

What does NOT live here
────────────────────────
  - Train/val/test splitting        → src/features/engineer.py
  - Scaling / imputation pipelines  → src/features/engineer.py
  - Any model-specific transforms   → src/models/
"""
from __future__ import annotations

import ast
import logging
import re
from typing import Sequence

import numpy as np
import pandas as pd

from configs.config import (
    COLUMNS_TO_DROP,
    POST_RELEASE_MODEL,
    POST_RELEASE_FEATURES,
    TARGET_LOG,
    TARGET_RAW,
    TOP_N_TAGS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    df: pd.DataFrame,
    *,
    post_release: bool = POST_RELEASE_MODEL,
    top_n_tags: int = TOP_N_TAGS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing sequence.  Returns a cleaned DataFrame ready for
    feature engineering.

    Parameters
    ----------
    df           : Raw merged DataFrame from loader.load_merged()
    post_release : If True, post-release columns (playtime, review counts, etc.)
                   are kept.  Set False for the launch-time prediction framing.
    top_n_tags   : How many of the most-frequent tags to one-hot encode.
    verbose      : Print step-by-step logs.
    """
    steps = [
        ("drop useless columns",        lambda d: drop_useless_columns(d)),
        ("drop target nulls",           lambda d: drop_target_nulls(d)),
        ("parse release date",          lambda d: parse_release_date(d)),
        ("parse supported languages",   lambda d: parse_language_count(d)),
        ("parse estimated owners",      lambda d: parse_estimated_owners(d)),
        ("parse genres",                lambda d: parse_genres(d)),
        ("parse categories",            lambda d: parse_categories(d)),
        ("parse tags",                  lambda d: parse_tags(d, top_n=top_n_tags)),
        ("derive numeric features",     lambda d: derive_numeric_features(d)),
        ("handle metacritic sparsity",  lambda d: handle_metacritic(d)),
        ("encode publisher class",      lambda d: encode_publisher_class(d)),
        ("frequency-encode dev/pub",    lambda d: frequency_encode_dev_pub(d)),
        ("log-transform skewed cols",   lambda d: log_transform_skewed(d)),
        ("create log target",           lambda d: create_log_target(d)),
    ]

    if not post_release:
        steps.append(("drop post-release features", lambda d: drop_post_release(d)))

    for name, fn in steps:
        if verbose:
            logger.info("  → %s", name)
        df = fn(df)

    if verbose:
        logger.info("Preprocessing complete: %d rows × %d columns", *df.shape)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Column drops
# ─────────────────────────────────────────────────────────────────────────────

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that carry no predictive signal:
    - Near-entirely null (Movies, Score rank, Metacritic url, Reviews, Notes)
    - URL / media columns (Header image, steamUrl, Screenshots)
    - Privacy / contact fields (Website, Support url, Support email)
    - Zero-variance columns (unreleased → all False)
    - User score (99.97 % zeros, effectively missing)
    """
    to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop)
    logger.debug("Dropped %d useless columns.", len(to_drop))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Target null removal
# ─────────────────────────────────────────────────────────────────────────────

def drop_target_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ~9 rows where copiesSold is NaN (Gamalytic had no match)."""
    before = len(df)
    df = df.dropna(subset=[TARGET_RAW]).copy()
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with null copiesSold.", dropped)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Date parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_release_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse `firstReleaseDate` (ISO-8601 from Gamalytic) into:
      release_year   int
      release_month  int  (1-12, useful for seasonality effects)
      release_quarter int (1-4)

    `Release date` (Kaggle, "Dec 8, 2021") is kept as backup but not parsed
    separately — firstReleaseDate covers the same information in a cleaner format.
    """
    df = df.copy()

    # firstReleaseDate: "2021-12-08T05:00:00.000Z"
    parsed = pd.to_datetime(df["firstReleaseDate"], utc=True, errors="coerce")

    df["release_year"]    = parsed.dt.year.astype("Int16")
    df["release_month"]   = parsed.dt.month.astype("Int8")
    df["release_quarter"] = parsed.dt.quarter.astype("Int8")

    # Drop raw date strings; 'Release date' from Kaggle is now redundant
    df = df.drop(columns=["firstReleaseDate", "Release date"], errors="ignore")

    logger.debug("release_year null count: %d", df["release_year"].isna().sum())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Language count
# ─────────────────────────────────────────────────────────────────────────────

_LIST_PATTERN = re.compile(r"['\"]([^'\"]+)['\"]")


def _parse_list_string(s: str | float) -> list[str]:
    """
    Parse a Python list literal stored as a string, e.g.
    "['English', 'Spanish', 'French']"  →  ['English', 'Spanish', 'French']
    Returns [] for NaN / unparseable values.
    """
    if pd.isna(s) or not isinstance(s, str):
        return []
    try:
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass
    # Fallback: regex extraction
    return _LIST_PATTERN.findall(s)


def parse_language_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    `Supported languages` is stored as a Python list literal string.
    We extract:
      language_count  int  – number of supported text languages

    The raw column is then dropped (it cannot be used directly by sklearn).
    """
    df = df.copy()
    df["language_count"] = (
        df["Supported languages"]
        .apply(_parse_list_string)
        .apply(len)
        .astype("int16")
    )
    df = df.drop(columns=["Supported languages"], errors="ignore")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Estimated owners
# ─────────────────────────────────────────────────────────────────────────────

def parse_estimated_owners(df: pd.DataFrame) -> pd.DataFrame:
    """
    `Estimated owners` is a range string from Kaggle, e.g. "200000 - 500000".
    We compute the midpoint and log-transform it as a corroborating signal.

    ⚠️  LEAKAGE NOTE: This is a post-release crowd estimate that correlates
    strongly with copiesSold.  It is grouped with POST_RELEASE_FEATURES and
    excluded when post_release=False.  When included, treat it with caution
    and do not report it as an independent predictor.
    """
    df = df.copy()

    def _midpoint(s: str | float) -> float:
        if pd.isna(s) or s == "0 - 0":
            return np.nan
        parts = str(s).split(" - ")
        try:
            lo, hi = float(parts[0]), float(parts[1])
            return (lo + hi) / 2.0
        except (ValueError, IndexError):
            return np.nan

    df["estimated_owners_midpoint"] = (
        df["Estimated owners"].apply(_midpoint)
    )
    df = df.drop(columns=["Estimated owners"], errors="ignore")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Genre one-hot encoding
# ─────────────────────────────────────────────────────────────────────────────

# Genres present in the data (comma-separated, no surrounding brackets)
KNOWN_GENRES = [
    "Action", "Adventure", "Casual", "Early Access", "Free to Play",
    "Indie", "Massively Multiplayer", "RPG", "Racing", "Simulation",
    "Sports", "Strategy", "Violent", "Sexual Content", "Gore",
]

def parse_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    `Genres` field: "Action,Indie,Early Access"
    Produces binary columns  genre_action, genre_indie, …
    Null genres → all zeros.
    """
    df = df.copy()

    def _to_set(s: str | float) -> set[str]:
        if pd.isna(s):
            return set()
        return {g.strip() for g in str(s).split(",")}

    genre_sets = df["Genres"].apply(_to_set)

    for genre in KNOWN_GENRES:
        col = "genre_" + genre.lower().replace(" ", "_")
        df[col] = genre_sets.apply(lambda gs: int(genre in gs)).astype("int8")

    df = df.drop(columns=["Genres"], errors="ignore")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Category binary flags
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_FLAGS: dict[str, list[str]] = {
    "has_singleplayer":      ["Single-player"],
    "has_multiplayer":       ["Multi-player", "Online PvP"],
    "has_coop":              ["Co-op", "Online Co-op", "Local Co-Op",
                              "Full Co-op", "Shared/Split Screen Co-op"],
    "has_vr":                ["VR Supported", "VR Only"],
    "has_controller":        ["Full controller support",
                              "Partial Controller Support",
                              "Tracked Controller Support"],
    "has_achievements":      ["Steam Achievements"],
    "has_trading_cards":     ["Steam Trading Cards"],
    "has_workshop":          ["Steam Workshop"],
    "has_family_sharing":    ["Family Sharing"],
    "has_cloud_saves":       ["Steam Cloud"],
    "has_leaderboards":      ["Steam Leaderboards"],
    "has_remote_play":       ["Remote Play Together",
                              "Remote Play on Phone",
                              "Remote Play on TV"],
}

def parse_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    `Categories` field: "Single-player,Steam Achievements,Family Sharing"
    Produces a set of interpretable binary feature columns.
    """
    df = df.copy()

    cat_sets = df["Categories"].apply(
        lambda s: set() if pd.isna(s) else {c.strip() for c in str(s).split(",")}
    )

    for col_name, keywords in CATEGORY_FLAGS.items():
        df[col_name] = cat_sets.apply(
            lambda cs: int(bool(cs.intersection(keywords)))
        ).astype("int8")

    df = df.drop(columns=["Categories"], errors="ignore")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Tag encoding
# ─────────────────────────────────────────────────────────────────────────────

def parse_tags(df: pd.DataFrame, *, top_n: int = TOP_N_TAGS) -> pd.DataFrame:
    """
    `Tags` field: "Action,Puzzle,Indie,2D,Pixel Graphics,…"
    Steps:
      1. Count total tags per game → tag_count
      2. Find the `top_n` most-frequent tags across all games
      3. One-hot encode each of those tags → tag_<name> columns
      4. Drop the raw Tags column
    Tags are ~28 % null — treated as "no tags" (all zeros).
    """
    df = df.copy()

    def _split_tags(s: str | float) -> list[str]:
        if pd.isna(s):
            return []
        return [t.strip() for t in str(s).split(",") if t.strip()]

    tag_lists = df["Tags"].apply(_split_tags)
    df["tag_count"] = tag_lists.apply(len).astype("int16")

    # Identify top-N tags by frequency
    from collections import Counter
    counter: Counter = Counter()
    for tags in tag_lists:
        counter.update(tags)
    top_tags = [tag for tag, _ in counter.most_common(top_n)]

    logger.info("Top %d tags (by frequency): %s", top_n, top_tags[:10])

    tag_sets = tag_lists.apply(set)
    for tag in top_tags:
        col = "tag_" + re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
        df[col] = tag_sets.apply(lambda ts: int(tag in ts)).astype("int8")

    df = df.drop(columns=["Tags"], errors="ignore")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Derived numeric features
# ─────────────────────────────────────────────────────────────────────────────

def derive_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engineered features from existing columns:

    total_reviews   = Positive + Negative
    review_ratio    = Positive / (total_reviews + 1)    # Wilson-style dampener
    is_free_to_play = 1 if Price == 0 else 0
    platform_count  = Windows + Mac + Linux  (number of supported platforms)
    description_length = character count of 'About the game' (proxy for dev effort)
    required_age_flag  = 1 if Required age > 0 else 0   (content flag)
    has_dlc            = 1 if DLC count > 0 else 0
    """
    df = df.copy()

    # Review-derived features
    df["total_reviews"] = (df["Positive"].fillna(0) + df["Negative"].fillna(0)).astype("int64")
    df["review_ratio"]  = (
        df["Positive"].fillna(0) / (df["total_reviews"] + 1)
    ).round(4)

    # Pricing
    df["is_free_to_play"] = (df["Price"] == 0).astype("int8")

    # Platform
    for col in ["Windows", "Mac", "Linux"]:
        df[col] = df[col].astype(bool).astype("int8")
    df["platform_count"] = (df["Windows"] + df["Mac"] + df["Linux"]).astype("int8")

    # Description length as a rough effort/quality signal
    df["description_length"] = (
        df["About the game"].fillna("").apply(len).astype("int32")
    )
    df = df.drop(columns=["About the game"], errors="ignore")

    # Age-rating flag
    df["required_age_flag"] = (df["Required age"].fillna(0) > 0).astype("int8")

    # DLC existence
    df["has_dlc"] = (df["DLC count"].fillna(0) > 0).astype("int8")

    # earlyAccess: cast bool string → int8
    df["earlyAccess"] = df["earlyAccess"].astype(bool).astype("int8")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — Metacritic sparsity
# ─────────────────────────────────────────────────────────────────────────────

def handle_metacritic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Metacritic score is 0 for ~96 % of games (not rated, not just bad).
    We add a binary indicator `has_metacritic` so models can distinguish
    "not rated" from a true score.  The original column is kept as-is.
    """
    df = df.copy()
    df["has_metacritic"] = (df["Metacritic score"].fillna(0) > 0).astype("int8")
    df["Metacritic score"] = df["Metacritic score"].fillna(0).astype("float32")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 11 — Publisher class ordinal encoding
# ─────────────────────────────────────────────────────────────────────────────

PUBLISHER_CLASS_ORDER = {"Hobbyist": 0, "Indie": 1, "AA": 2, "AAA": 3}

def encode_publisher_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    publisherClass is an ordinal categorical: Hobbyist < Indie < AA < AAA.
    Encode as 0-3; unknown / null → -1 so models can detect missingness.
    """
    df = df.copy()
    df["publisher_class_ord"] = (
        df["publisherClass"]
        .map(PUBLISHER_CLASS_ORDER)
        .fillna(-1)
        .astype("int8")
    )
    df = df.drop(columns=["publisherClass"], errors="ignore")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 12 — Developer / Publisher frequency encoding
# ─────────────────────────────────────────────────────────────────────────────

def frequency_encode_dev_pub(df: pd.DataFrame) -> pd.DataFrame:
    """
    Developers and Publishers are high-cardinality text columns.
    We use frequency encoding: replace each value with the number of games
    in the dataset that share that developer/publisher.  This is a leakage-safe
    encoding (uses the full dataset frequency, not a target statistic).

    Null → 0 (unknown / no publisher listed).
    """
    df = df.copy()

    for raw_col, new_col in [("Developers", "developer_freq"),
                              ("Publishers", "publisher_freq")]:
        freq_map = df[raw_col].fillna("Unknown").value_counts().to_dict()
        df[new_col] = (
            df[raw_col].fillna("Unknown").map(freq_map).fillna(0).astype("int32")
        )
        df = df.drop(columns=[raw_col], errors="ignore")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 13 — Log-transform heavily skewed numeric columns
# ─────────────────────────────────────────────────────────────────────────────

LOG1P_COLS = [
    "Price",
    "Positive",
    "Negative",
    "total_reviews",
    "Recommendations",
    "Peak CCU",
    "Average playtime forever",
    "Average playtime two weeks",
    "Median playtime forever",
    "Median playtime two weeks",
    "DLC count",
    "Achievements",
    "estimated_owners_midpoint",
]

def log_transform_skewed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to heavily right-skewed numeric columns.
    The original columns are replaced; the new values are named consistently.
    Columns not present in the DataFrame are silently skipped.
    """
    df = df.copy()
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0).fillna(0)).astype("float32")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 14 — Log-transform the target
# ─────────────────────────────────────────────────────────────────────────────

def create_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `log_copies_sold` = log1p(copiesSold).
    Both the raw and log targets are kept so the modelling layer can choose.
    """
    df = df.copy()
    df[TARGET_LOG] = np.log1p(df[TARGET_RAW]).astype("float32")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Post-release feature removal (launch-time framing)
# ─────────────────────────────────────────────────────────────────────────────

def drop_post_release(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are only observable after a game has launched.
    Called when post_release=False (launch-time prediction framing).
    """
    to_drop = [c for c in POST_RELEASE_FEATURES if c in df.columns]
    df = df.drop(columns=to_drop)
    logger.info("Launch-time mode: dropped %d post-release columns.", len(to_drop))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Utility: column audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame: column name, dtype, null %, nunique, sample values.
    Useful for sanity-checking after preprocessing.
    """
    rows = []
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        rows.append({
            "column":    col,
            "dtype":     str(df[col].dtype),
            "null_pct":  round(null_pct, 2),
            "nunique":   df[col].nunique(),
            "sample":    str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—",
        })
    return pd.DataFrame(rows).sort_values("null_pct", ascending=False)
