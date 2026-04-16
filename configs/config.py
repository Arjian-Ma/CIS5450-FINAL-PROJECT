"""
Central configuration for the Steam Sales Prediction project.
Edit paths and flags here; nothing else in the codebase hard-codes them.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "Data"
OUTPUT_DIR = ROOT_DIR / "outputs"   # created at runtime if needed

RAW_MERGED_PATH    = DATA_DIR / "games_merged.csv"
RAW_GAMES_PATH     = DATA_DIR / "games.csv"
RAW_GAMELIST_PATH  = DATA_DIR / "games-list.csv"

PROCESSED_PATH     = OUTPUT_DIR / "processed.parquet"
FEATURES_PATH      = OUTPUT_DIR / "features.parquet"

# ── Model framing ──────────────────────────────────────────────────────────────
# True  → post-release explanatory model (uses playtime, review counts, etc.)
# False → launch-time prediction model  (only pre-release / at-launch features)
POST_RELEASE_MODEL: bool = True

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET_RAW = "copiesSold"
TARGET_LOG = "log_copies_sold"   # log1p transform applied in feature engineering

# ── Data split ─────────────────────────────────────────────────────────────────
TEST_SIZE       = 0.15
VAL_SIZE        = 0.15   # fraction of the remaining train set used for validation
RANDOM_STATE    = 42

# ── Feature lists ──────────────────────────────────────────────────────────────
# Post-release metrics — include only when POST_RELEASE_MODEL is True.
# These are observed *after* a game launches; using them in a launch-time model
# would constitute target leakage.
POST_RELEASE_FEATURES = [
    "Positive",
    "Negative",
    "total_reviews",
    "review_ratio",
    "Recommendations",
    "Average playtime forever",
    "Average playtime two weeks",
    "Median playtime forever",
    "Median playtime two weeks",
    "Peak CCU",
    "estimated_owners_midpoint",   # Kaggle range estimate, also post-release
]

# Columns dropped before any modelling — URLs, near-constant, or irrelevant text
COLUMNS_TO_DROP = [
    "Movies",          # 100 % null
    "Score rank",      # 99.97 % null
    "Metacritic url",  # 96 % null, not predictive
    "Reviews",         # 89 % null, free-text press quotes
    "Notes",           # 80 % null, mostly empty
    "Website",         # 57 % null, not predictive
    "Support url",     # 54 % null
    "Support email",   # not predictive
    "Header image",    # URL
    "steamUrl",        # URL
    "Screenshots",     # URLs
    "User score",      # 99.97 % zero (effectively all missing)
    "unreleased",      # all False — no variance
    "Full audio languages",  # very sparse; language_count covers the signal
]

# ── Tag encoding ───────────────────────────────────────────────────────────────
# Number of most-frequent tags to one-hot encode (rest are dropped)
TOP_N_TAGS = 30

# ── Model hyper-parameters (defaults) ─────────────────────────────────────────
RIDGE_ALPHA  = 1.0
LASSO_ALPHA  = 0.01

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="rmse",
    random_state=RANDOM_STATE,
)
