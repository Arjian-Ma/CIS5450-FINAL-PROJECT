# RFE + Cross-Validation: How Feature Selection Works in This Project
**CIS 5450 Final Project — Steam Sales Prediction**

---

## Table of Contents

1. [Setup](#1-setup)
2. [How RFE + CV Works (the actual mechanics)](#2-how-rfe--cv-works-the-actual-mechanics)
3. [Why 71 Features Won Over 70 (and the 1-SE Rule Picked 60)](#3-why-71-features-won-over-70-and-the-1-se-rule-picked-60)
4. [Visualising What Happened](#4-visualising-what-happened)
5. [Practical Takeaways](#5-practical-takeaways)
6. [Reproducibility](#6-reproducibility)

---

## 1. Setup

| Item | Value |
|---|---|
| Script | [`run_rfe.py`](../run_rfe.py) |
| Total feature pool | 79 (after `use_release_year=False`, `use_game_age=True`) |
| Estimator inside RFE | XGBoost, `tree_method="hist"`, `n_estimators=200` |
| `step` | 2 (drop 2 features per RFE round) |
| `cv` | 3 (3-fold cross-validation) |
| `min_features` | 5 |
| Subset sizes evaluated | 79, 77, 75, …, 7, 5 → **38 sizes total** |

Two distinct picks come out of the run:

- **`chosen_by_min` = 71 features** — lowest mean CV RMSE
- **`chosen_by_1se` = 60 features** — smallest subset within 1 std-err of the best (the "1-SE rule")

CV results are saved to [`outputs/rfe/rfecv_selected_cv_table.csv`](../outputs/rfe/rfecv_selected_cv_table.csv); ranking is in [`outputs/rfe/rfecv_selected_ranking.csv`](../outputs/rfe/rfecv_selected_ranking.csv); CV-RMSE-vs-#-features plot is at [`outputs/rfe/rfecv_selected_curve.png`](../outputs/rfe/rfecv_selected_curve.png).

---

## 2. How RFE + CV Works (the actual mechanics)

### Step 1 — Plan the sweep

Given 79 starting features, `step=2`, and `min_features=5`, RFECV will evaluate every subset size from 79 down to 5 in steps of 2. That's 38 subset sizes.

### Step 2 — At each subset size, do 3-fold CV

The training set has 80,633 rows. Split it into 3 equal folds (A, B, C). For each subset size `k`:

1. Train XGBoost on **A+B** (~54,422 rows), predict on **C** (~26,211 rows) → record RMSE on the log target.
2. Train on **A+C**, predict on **B** → record RMSE.
3. Train on **B+C**, predict on **A** → record RMSE.
4. Record `mean_rmse` and `std_rmse` across the 3 folds.

This populates the `split0_cv_rmse_log`, `split1_cv_rmse_log`, `split2_cv_rmse_log` columns of the CV table. Each row is "what happens at this subset size, across the 3 folds."

### Step 3 — How RFE decides which features to drop at each round

Inside RFECV, the elimination is sequential. At round 0 the model has all 79 features:

1. Train one XGBoost on the full training set (no CV — just one fit).
2. Read `feature_importances_` from that model — gain-based.
3. Drop the **bottom 2** features by importance.

Now there are 77 features. Repeat the cycle: train, read importances, drop bottom 2 → 75 features. And so on, all the way down to 5.

The CV from Step 2 happens *in parallel* with this — at each subset size we evaluate "if I stopped here, how good is the model?" via 3-fold CV. The eliminations themselves are deterministic once you fix the random seed.

### Step 4 — Pick the winner

After all 38 subset sizes are evaluated, RFECV looks at the table and picks the size with the lowest mean CV RMSE. That's `chosen_by_min` (= 71 features here).

The **1-SE rule** (a separate column we added to the CV table) does:

1. Take the best mean RMSE (`1.55219` at n=71)
2. Take its std (`0.00351`)
3. Threshold = `1.55219 + 0.00351 = 1.55570`
4. Find the smallest n where mean RMSE ≤ 1.55570 → **n = 60** (mean = 1.55485)

In words: *the smallest subset whose CV RMSE is within one standard error of the best.*

---

## 3. Why 71 Features Won Over 70 (and the 1-SE Rule Picked 60)

Looking at the actual numbers in the 60-79 feature range:

| n_features | mean_rmse | std_rmse |
|---:|---:|---:|
| 60 | 1.55485 | 0.00393 |
| 65 | 1.55528 | 0.00356 |
| 66 | 1.55331 | 0.00390 |
| 67 | 1.55547 | 0.00467 |
| 68 | 1.55400 | 0.00297 |
| 69 | 1.55346 | 0.00432 |
| 70 | 1.55365 | 0.00414 |
| **71** | **1.55219** | 0.00351 ← winner |
| 72 | 1.55465 | 0.00149 |
| 73 | 1.55515 | 0.00452 |
| 74 | 1.55426 | 0.00459 |
| 75 | 1.55502 | 0.00334 |
| 76 | 1.55547 | 0.00381 |
| 77 | 1.55224 | 0.00315 ← runner-up |
| 78 | 1.55552 | 0.00200 |
| 79 | 1.55390 | 0.00129 |

The **gap between 71 and 70** is `1.55365 - 1.55219 = 0.00146`. That's the entire margin of victory.

**Compare that gap to the std at n=71**: std = `0.00351`. The "winning" gap is **less than half the within-fold noise**. If you re-ran with a different random seed for the CV folds, 70 (or 60, or 77) could easily come out on top.

**This is exactly why the 1-SE rule exists.** It says: "the gap from 71 → 60 is `1.55485 - 1.55219 = 0.00266`, which is smaller than 71's std of 0.00351 — so 60 features are *statistically indistinguishable* from 71, and you should prefer the smaller model."

---

## 4. Visualising What Happened

If you plot the curve from 5 to 79 features (see [`outputs/rfe/rfecv_selected_curve.png`](../outputs/rfe/rfecv_selected_curve.png)), three regions are visible:

| Region | n_features | Mean RMSE behaviour | Interpretation |
|---|---|---|---|
| Steep | 5 → 15 | 1.91 → 1.60 | Each added feature genuinely adds signal |
| Diminishing returns | 15 → 40 | 1.60 → 1.56 | Marginal predictors fading in slowly |
| **Noise floor** | 40 → 79 | ~1.555 ± 0.003 | Curve wiggles but doesn't trend |

71 winning over 70 is just the curve's last micro-wiggle landing in 71's favour. It's not "71 has some special structural property" — it's stochastic noise at the bottom of an L-curve.

---

## 5. Practical Takeaways

1. **The "best" subset count is essentially arbitrary above ~40 features.** Anywhere from 40 to 79 features performs within noise of each other. The model doesn't actually need 71 features.

2. **You should genuinely prefer the 1-SE pick (60) for the writeup.** It's smaller, it's statistically tied with 71, and *"we use 60 features chosen by RFECV with the 1-SE rule"* is more defensible than *"we use 71 because the curve had a tiny dip there."*

3. **The error analysis using top-60 was the right call.** Same model quality (within CV noise) with 11 fewer features, cleaner story.

4. **What CV is doing here** is *robustness checking*, not selection: it answers "if I trained on a different chunk of training data, would this feature subset still look good?" — without it, the score for "best 30 features" would be based on one fit and you couldn't tell if 30 vs 28 features was a real difference or random luck.

---

## 6. Reproducibility

```bash
# Re-run the full RFECV sweep
python run_rfe.py

# Faster version (step=5, cv=3) for iteration
python run_rfe.py --step 5

# Higher-stability version (cv=5) — slower
python run_rfe.py --cv 5
```

Outputs land at:

- [`outputs/rfe/rfecv_selected_ranking.csv`](../outputs/rfe/rfecv_selected_ranking.csv) — feature-by-feature ranking + final-model importances
- [`outputs/rfe/rfecv_selected_cv_table.csv`](../outputs/rfe/rfecv_selected_cv_table.csv) — per-subset-size CV scores; this is what the analysis above is built from
- [`outputs/rfe/rfecv_selected_curve.png`](../outputs/rfe/rfecv_selected_curve.png) — CV-RMSE-vs-#-features plot
