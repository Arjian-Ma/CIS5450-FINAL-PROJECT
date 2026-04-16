# Data Findings, Preprocessing Design & Code Workflow
**CIS 5450 Final Project — Steam Sales Prediction**

---

## Table of  sb

1. [Dataset Overview](#1-dataset-overview)
2. [Data Findings by Column Group](#2-data-findings-by-column-group)
3. [Critical Data Quality Issues](#3-critical-data-quality-issues)
4. [Preprocessing Design: Every Decision Explained](#4-preprocessing-design-every-decision-explained)
5. [Feature Engineering Overview](#5-feature-engineering-overview)
6. [Leakage Map](#6-leakage-map)
7. [Code Workflow: How to Run the Project](#7-code-workflow-how-to-run-the-project)
8. [What Is Implemented vs. TODO](#8-what-is-implemented-vs-todo)

---

## 1. Dataset Overview

### Source files

| File | Rows | Source | Role |
|---|---|---|---|
| `Data/games-list.csv` | 117,629 | Gamalytic | Sales estimates, publisher classification |
| `Data/games.csv` | ~115,000 | Kaggle Steam | Game listing metadata |
| `Data/games_merged.csv` | **115,289** | Merged | Primary working file — 47 columns |

The merge is a **left join** on `AppID = steamId`. Almost every Kaggle game matched a Gamalytic record (only 9 rows have a null target after merging).

### After preprocessing

The pipeline produces **97 columns** from the original 47, primarily because `Genres`, `Categories`, and `Tags` are exploded into binary feature columns.

---

## 2. Data Findings by Column Group

### 2.1 Target variable: `copiesSold`

This is the single most important finding about the data.

- **Skewness ≈ 70** — one of the most extreme right-skewed distributions possible in a real dataset.
- The median game sold **613 copies**. The top 1% sold over **935,000**.
- Counter-Strike 2 (the top game) sold **343 million** copies — roughly **560,000× the median**.

```
Percentile breakdown:
  50th:    613 copies
  75th:  4,285 copies
  90th: 30,888 copies
  95th: 116,920 copies
  99th: 935,298 copies
```

**Implication**: You cannot model raw `copiesSold` with linear regression — a single blockbuster will dominate the loss. Always use `log1p(copiesSold)` as the target. Report metrics on both scales but use the log scale for model comparisons.

---

### 2.2 Publisher class (Gamalytic)

A direct label from Gamalytic classifying the publishing entity.

```
Hobbyist    66,045  (56%)   ← solo devs / hobby projects
Indie       46,949  (40%)   ← small independent studios
AA           3,313   (3%)   ← mid-tier publishers
AAA          1,322   (1%)   ← major publishers (Valve, EA, Ubisoft …)
```

**Finding**: Publisher class is one of the strongest signals in the data. Median sales per class span roughly **4 orders of magnitude** — a AAA game's median is ~100× a Hobbyist game's median. This is almost certainly the single most predictive categorical feature.

---

### 2.3 Price

Two price columns exist — from different sources:

| Column | Source | Units | Range |
|---|---|---|---|
| `Price` | Kaggle | USD | 0 – 290 |
| `price` | Gamalytic | USD | 0 – 1,900 (not in merged) |

The Gamalytic `price` column was **not included in the merge** (only `copiesSold`, `reviewScore`, `publisherClass`, `earlyAccess`, `firstReleaseDate`, `unreleased`, `steamUrl` were kept). Use the Kaggle `Price` column.

Key findings:
- **14.6% of games are free** (`Price == 0`).
- Free games have a **~5× higher median copiesSold** than paid games (2,562 vs. 421 in a 10k sample).
- Price distribution is right-skewed: most paid games are $0.99–$19.99; a small tail goes up to $290.
- The TA noted that Gamalytic estimates for free games are noisier — treat the F2P vs. paid comparison as exploratory.

---

### 2.4 Review signals

| Column | Finding |
|---|---|
| `reviewScore` (Gamalytic, 0–100) | Good coverage across all games; mean=72, std=29 |
| `Positive` (Kaggle review count) | Heavily zero-inflated — 31% of games in a 10k sample have zero positive reviews |
| `Negative` (Kaggle review count) | Similarly sparse |
| `Recommendations` | 75th percentile is 0 — most small games have no recommendations |
| `User score` | **99.97% zeros** — essentially missing data, not a real score |
| `Metacritic score` | **96% zeros** — most games are not on Metacritic |

`reviewScore` (Gamalytic's aggregate) is the most usable review feature. `Positive` and `Negative` are useful as derived features (`total_reviews`, `review_ratio`) after log-transforming, but they are **post-release signals** and must be handled as such.

---

### 2.5 Genres, Categories, Tags

All three are stored as raw comma-separated strings. They must be parsed before use.

**Genres** (0.2% null — very clean):
```
Example: "Action,Indie,Early Access"
Top genres: Indie (55%), Action (30%), Casual (25%), Adventure (18%), Simulation (15%)
```

**Categories** (essentially no nulls):
```
Example: "Single-player,Steam Achievements,Family Sharing"
Most games are singleplayer; multiplayer tags are much rarer.
```

**Tags** (28.6% null — treat nulls as "no tags"):
```
Example: "Action,Puzzle,Indie,2D,Pixel Graphics,Atmospheric"
Top tags by frequency: Singleplayer, Indie, Action, Casual, Adventure, 2D, 3D, Strategy
```

Tags are the richest categorical signal (up to 20 tags per game). They are more granular than genres.

---

### 2.6 Date features

Two date columns, both encoding the same information:

| Column | Format | Source |
|---|---|---|
| `Release date` | "Dec 8, 2021" | Kaggle |
| `firstReleaseDate` | "2021-12-08T05:00:00.000Z" | Gamalytic |

`firstReleaseDate` is used because it's in ISO 8601 format and easier to parse reliably. Both columns are dropped after extracting `release_year`, `release_month`, and `release_quarter`.

**Finding**: The number of games released per year has grown sharply — 2022–2024 have far more games than earlier years. This matters for modelling: newer cohorts have less time to accumulate sales. Consider whether to include `release_year` as a raw feature or compute "game age in days."

---

### 2.7 Supported languages

Stored as a Python list literal string:
```
"['Simplified Chinese', 'Japanese', 'English']"  →  language_count = 3
```

The distribution is right-skewed: most indie games support 1–3 languages; large games support 20+. Language count correlates weakly but positively with sales.

---

### 2.8 Columns with near-total missingness (dropped)

| Column | Null % | Reason for dropping |
|---|---|---|
| `Movies` | 100% | Completely empty |
| `Score rank` | 99.97% | Only 3 non-null values across 115k rows |
| `Metacritic url` | 96.2% | URL; no predictive value |
| `Reviews` | 89.4% | Free-text press quotes — not structured |
| `Notes` | 80.3% | Miscellaneous admin notes |
| `Website` | 57.2% | URL; not predictive |
| `Support url` | 53.5% | URL; not predictive |
| `User score` | ~99.97% zero | Functionally all missing |

---

### 2.9 Early access and platform flags

- **11,204 games** (9.5%) were tagged as Early Access at some point.
- **`Windows`**: 99.98% True — almost no variance. Kept but rarely contributes.
- **`Mac`**: 18.75% True — a meaningful minority.
- **`Linux`**: 14.09% True — smaller minority.
- `platform_count` = Windows + Mac + Linux captures multi-platform reach in one feature.

---

## 3. Critical Data Quality Issues

### Issue 1: `copiesSold` is an estimate, not ground truth
Gamalytic derives sales from review counts and a conversion model. The estimates are **more reliable for mid-to-large games** and **less reliable for small free-to-play games** (which can have very high download counts but few reviews). Your write-up must acknowledge this uncertainty — use "estimated sales" not "sales."

### Issue 2: Two overlapping price columns
Gamalytic's `price` was excluded from the merge. The Kaggle `Price` is the correct column to use. They represent the same thing (current price in USD) but may diverge for games with price changes over time.

### Issue 3: `Estimated owners` vs `copiesSold`
`Estimated owners` (Kaggle) is a **coarse band** (e.g., "200,000 – 500,000") derived from a similar estimation model. It is strongly correlated with `copiesSold`. **Using it as a feature in a model that predicts `copiesSold` borders on leakage** — both are estimates of the same underlying truth. It is classified as a `POST_RELEASE_FEATURE` and excluded in launch-time mode. If included, flag it explicitly in your report.

### Issue 4: Post-release feature leakage
`Positive`, `Negative`, `Recommendations`, `Average playtime`, `Peak CCU`, and `Estimated owners` are only observable after a game has been live. If your goal is to predict success from the Steam listing at launch, these must be excluded. The codebase supports both framings via the `POST_RELEASE_MODEL` config flag.

---

## 4. Preprocessing Design: Every Decision Explained

The pipeline runs 14 steps in order (see [`src/data/preprocessor.py`](../src/data/preprocessor.py)).

### Step 1 — Drop useless columns
Drop columns with ≥80% nulls, URL-only content, zero variance (`unreleased` is all False), or that are functionally empty (`User score`). This removes 14 columns before any parsing begins.

### Step 2 — Drop target nulls
9 rows have `NaN` for `copiesSold` (Gamalytic had no match). These are dropped. Any future re-merge should produce the same 9 or fewer nulls.

### Step 3 — Parse release date
`firstReleaseDate` (ISO 8601) → `release_year`, `release_month`, `release_quarter`. The raw date columns are dropped. Year captures the cohort effect; month captures seasonality (Q4 releases — October through December — tend to sell more during holiday periods).

### Step 4 — Parse language count
`Supported languages` is a Python list literal stored as a string. It is parsed with `ast.literal_eval()` (with a regex fallback) → integer `language_count`. The raw column is dropped since sklearn cannot use list strings directly.

### Step 5 — Parse estimated owners
`Estimated owners` range string → midpoint float. Classified as post-release. Null (the "0 - 0" band for games with no detectable owners) is set to `NaN` rather than 0, because 0 is a meaningful lower estimate.

### Step 6 — Genres → binary columns
Top genres are one-hot encoded as `genre_action`, `genre_indie`, etc. Null genres → all zeros. Using binary columns rather than a single multi-class encoding allows models to correctly represent multi-genre games (e.g., "Action,Indie" is both `genre_action=1` AND `genre_indie=1`).

### Step 7 — Categories → binary flags
Rather than one-hot encoding every possible category string, we define 12 interpretable flags: `has_singleplayer`, `has_multiplayer`, `has_coop`, `has_vr`, `has_controller`, `has_achievements`, `has_trading_cards`, `has_workshop`, `has_family_sharing`, `has_cloud_saves`, `has_leaderboards`, `has_remote_play`. These capture the business-meaningful distinctions.

### Step 8 — Tags → binary columns + count
1. Count total tags per game → `tag_count` (rough signal of how much metadata effort went into the listing).
2. Find the top-30 most frequent tags across all 115k games.
3. One-hot encode each → `tag_singleplayer`, `tag_indie`, `tag_action`, etc.
4. Drop the raw `Tags` column. The 28.6% null rows get all-zero tag columns and `tag_count = 0`.

The number of top tags to encode is controlled by `TOP_N_TAGS` in `configs/config.py`.

### Step 9 — Derived numeric features
| New feature | Formula | Rationale |
|---|---|---|
| `total_reviews` | Positive + Negative | Raw review volume |
| `review_ratio` | Positive / (total_reviews + 1) | Quality signal; +1 prevents division by zero |
| `is_free_to_play` | Price == 0 | Explicit F2P indicator |
| `platform_count` | Windows + Mac + Linux | Multi-platform reach |
| `description_length` | len(About the game) | Proxy for developer effort / listing quality |
| `required_age_flag` | Required age > 0 | Mature content indicator |
| `has_dlc` | DLC count > 0 | DLC existence (separates from count scale) |

### Step 10 — Metacritic sparsity
96% of games have a Metacritic score of 0 — but 0 does NOT mean "zero score"; it means "not rated." Adding `has_metacritic = 1` for the 4% that are rated lets models use the score correctly. The original `Metacritic score` column is kept as-is with 0 meaning "unrated."

### Step 11 — Publisher class ordinal encoding
`publisherClass` has a natural order: Hobbyist < Indie < AA < AAA. It is encoded as integers 0–3. Unknown/null → -1 (so models can detect a missing publisher class rather than confusing it with Hobbyist=0).

### Step 12 — Developer/Publisher frequency encoding
`Developers` and `Publishers` can have thousands of unique values. Target encoding (mean sales per developer) would cause leakage. **Frequency encoding** — replacing each value with the count of games from that entity in the dataset — is leakage-safe and captures the signal that prolific developers/publishers tend to have higher sales.

### Step 13 — Log-transform skewed numeric columns
`Price`, `Positive`, `Negative`, `total_reviews`, `Recommendations`, `Peak CCU`, `Average playtime`, `DLC count`, `Achievements` are all heavily right-skewed. `log1p()` is applied to all of them in-place. This makes linear models more effective and prevents a single blockbuster value from dominating tree splits.

### Step 14 — Create log target
`log_copies_sold = log1p(copiesSold)`. Both the raw and log targets are kept in the DataFrame so the modelling layer can choose.

---

## 5. Feature Engineering Overview

After preprocessing, `src/features/engineer.py` builds the final sklearn-compatible feature matrix.

**Feature counts:**
- Post-release model: **~93 features**
- Launch-time model: **~82 features** (post-release signals removed)

**Feature categories:**

| Category | Examples | Count |
|---|---|---|
| Continuous (log-transformed) | Price, Positive, Negative, total_reviews, Average playtime, Recommendations | ~15 |
| Date | release_year, release_month, release_quarter | 3 |
| Genre binary | genre_action, genre_indie, genre_rpg … | 15 |
| Category flags | has_singleplayer, has_multiplayer, has_coop … | 12 |
| Tag binary | tag_singleplayer, tag_indie, tag_action … | 30 |
| Derived | is_free_to_play, platform_count, description_length, review_ratio … | ~8 |
| Developer/Publisher | developer_freq, publisher_freq | 2 |
| Publisher class | publisher_class_ord | 1 |
| Metadata-derived | language_count, tag_count, has_metacritic, Metacritic score | 4 |

**sklearn ColumnTransformer** (fit on train only, applied to val/test):
- **Continuous columns**: median imputation → StandardScaler
- **Binary/ordinal columns**: most-frequent imputation → passthrough (no scaling)

---

## 6. Leakage Map

This is the most important thing to get right in the write-up.

| Feature | Leakage risk | Available at launch? | Included in launch-time model? |
|---|---|---|---|
| `Price` | None | Yes (set at launch) | Yes |
| `publisher_class_ord` | None | Yes | Yes |
| `release_year/month` | None | Yes | Yes |
| `genre_*`, `has_*`, `tag_*` | None | Yes | Yes |
| `language_count` | None | Yes | Yes |
| `earlyAccess` | None | Yes | Yes |
| `description_length` | None | Yes | Yes |
| `Metacritic score` | Mild — only available if game was reviewed | Yes if rated | Yes |
| `reviewScore` (Gamalytic) | **Post-release** — aggregated over lifetime | No | No |
| `Positive`, `Negative`, `total_reviews` | **Post-release** | No | No |
| `Recommendations` | **Post-release** | No | No |
| `Average playtime`, `Median playtime` | **Post-release** | No | No |
| `Peak CCU` | **Post-release** | No | No |
| `estimated_owners_midpoint` | **Post-release** + same-target proxy | No | No |

---

## 7. Code Workflow: How to Run the Project

### Install dependencies
```bash
pip install -r requirements.txt
```

### Option A: Run the full pipeline from the CLI
```bash
# Post-release explanatory model (all features), baselines only
python main.py --framing post_release --models baseline

# Launch-time prediction model, all models including XGBoost
python main.py --framing launch --models all

# Save processed data + feature arrays to outputs/
python main.py --save

# Print column audit after preprocessing
python main.py --audit
```

### Option B: Run notebooks in order
The notebooks in `notebooks/` are written as `.py` files with `# %%` cell markers. They can be opened in:
- **VS Code** with the Jupyter extension (run cells with Shift+Enter)
- **JupyterLab** via `jupytext` (converts `.py` ↔ `.ipynb`)
- **Directly** as Python scripts: `python notebooks/01_eda.py`

Run them in order:
```
01_eda.py              → understand the raw data
02_preprocessing.py    → verify cleaning steps, save processed.parquet
03_modeling.py         → fit and compare all models
04_hypothesis_testing.py → run the three statistical tests
```

All notebooks expect `outputs/processed.parquet` to exist. Run notebook 02 (or `python main.py --save`) before notebooks 03 and 04.

### Option C: Use the modules directly in your own code
```python
import sys
sys.path.insert(0, ".")   # add project root to path

from configs.config import RAW_MERGED_PATH, POST_RELEASE_MODEL
from src.data.loader import load_merged, validate_merged
from src.data.preprocessor import run_preprocessing_pipeline
from src.features.engineer import prepare_features
from src.models.baseline import run_all_baselines
from src.models.advanced import XGBoostModel
from src.evaluation.metrics import evaluate_predictions

# Load
df_raw = load_merged(RAW_MERGED_PATH)
validate_merged(df_raw)

# Preprocess
df = run_preprocessing_pipeline(df_raw, post_release=True)

# Feature matrix + split
data = prepare_features(df, post_release=True)

# Baselines
summary = run_all_baselines(data)

# Best advanced model
xgb = XGBoostModel()
xgb.fit(data["X_train"], data["y_train"])
val_metrics = xgb.evaluate(data["X_val"], data["y_val"], "val")
```

---

### Key config flags in `configs/config.py`

| Flag | Default | Effect |
|---|---|---|
| `POST_RELEASE_MODEL` | `True` | Switch between post-release and launch-time framing |
| `TOP_N_TAGS` | `30` | How many top tags to one-hot encode |
| `RIDGE_ALPHA` | `1.0` | Ridge regularisation strength |
| `LASSO_ALPHA` | `0.01` | Lasso regularisation strength |
| `TEST_SIZE` | `0.15` | Fraction of data held out as test set |
| `VAL_SIZE` | `0.15` | Fraction of remaining data used for validation |

---

## 8. What Is Implemented vs. TODO

### Fully implemented
- Data loading and validation (`src/data/loader.py`)
- All 14 preprocessing steps (`src/data/preprocessor.py`)
- Feature matrix construction, stratified splitting, sklearn pipeline (`src/features/engineer.py`)
- Evaluation on log and raw scales (`src/evaluation/metrics.py`)
- **Baseline models**: MeanPredictor, LinearRegression, Ridge, Lasso (`src/models/baseline.py`)
- Lasso feature importance analysis
- Advanced model class architecture with `fit` / `predict` / `evaluate` / `feature_importance_df` (`src/models/advanced.py`)
- Hypothesis test scaffolding: correct signatures, docstrings, guard clauses, `HypothesisResult` dataclass (`src/hypothesis/tests.py`)
- Notebooks 01–04 (cells structured and annotated; models cells call implemented src functions)
- CLI pipeline runner (`main.py`)

### TODO — to be filled in

| File | Location | Task |
|---|---|---|
| `src/hypothesis/tests.py` | `test_free_vs_paid()` body | Implement permutation loop + bootstrap CI |
| `src/hypothesis/tests.py` | `test_price_effect()` body | Implement OLS permutation test |
| `src/hypothesis/tests.py` | `test_publisher_class()` body | Implement permutation F-test |
| `src/hypothesis/tests.py` | `bootstrap_ci()` | Implement bootstrap loop |
| `src/models/advanced.py` | `XGBoostModel.fit()` | Add early stopping (code commented in-place) |
| `src/models/advanced.py` | All model classes | Hyperparameter tuning (guidance in TODO comments) |
| `notebooks/03_modeling.py` | SHAP section | Add SHAP analysis after best model is chosen |
| `notebooks/03_modeling.py` | Test set cell | Uncomment and run only once model is finalised |

---

*Document last updated: April 2026*
