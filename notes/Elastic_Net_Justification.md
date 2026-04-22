# Why Elastic Net for PHQ-9 Regression on eGeMAPS Features

**TL;DR.** We have n=52 participants and 88 eGeMAPS features that include many highly correlated groups (F0 percentiles among themselves, loudness percentiles, voiced-vs-full MFCC pairs). Plain OLS is ill-conditioned; pure LASSO is unstable under correlated features; pure Ridge can't do feature selection. **Elastic Net does both regularization and selection, and handles correlated features coherently** — which is why it's the right default here.

---

## 1. The three regularized linear regressions — terminology

All three share the same basic setup. We're fitting `ŷ = β₁·x₁ + β₂·x₂ + ...` and minimizing some loss.

- **Ridge regression** = L2 penalty only (`λ · Σβ²`). Shrinks coefficients toward zero, but never exactly to zero. No feature selection.
- **LASSO** = L1 penalty only (`λ · Σ|β|`). Shrinks *and* zeros out coefficients → does feature selection.
- **Elastic Net** = both combined. This is what people usually mean when they informally say "ridge + LASSO."

---

## 2. Concrete worked example

### The data

3 observations, 2 features. Call them `x₁` and `x₂`, target `y`.

| i | x₁ | x₂ | y  |
|---|----|----|----|
| 1 | 1  | 1  | 2  |
| 2 | 2  | 3  | 7  |
| 3 | 3  | 5  | 12 |

We want to fit `ŷ = β₁·x₁ + β₂·x₂`.

### Step 1 — Write out RSS explicitly

**RSS** (residual sum of squares) = "sum of (actual minus predicted) squared, over all data points." For our 3 rows:

```
RSS(β₁, β₂) = (y₁ - ŷ₁)² + (y₂ - ŷ₂)² + (y₃ - ŷ₃)²
            = (2  - β₁  - β₂ )²
            + (7  - 2β₁ - 3β₂)²
            + (12 - 3β₁ - 5β₂)²
```

That's **the** loss function for plain OLS. Find the β₁, β₂ that make this smallest.

For this data, plain OLS gives **β_OLS = (−1, 3)**. Plug in to check:

- ŷ₁ = −1·1 + 3·1 = 2  ✓
- ŷ₂ = −1·2 + 3·3 = 7  ✓
- ŷ₃ = −1·3 + 3·5 = 12 ✓

So RSS(−1, 3) = 0. Perfect fit — but that coefficient of −1 is suspicious (why would feature 1 *hurt* the prediction?). This is exactly the kind of wild coefficient regularization is meant to tame.

### Step 2 — Ridge: add the L2 penalty to RSS

Ridge's loss function is:

```
L_ridge(β₁, β₂) = RSS(β₁, β₂)  +  λ · (β₁² + β₂²)
                  ^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^
                  same as OLS     the "penalty"
```

The penalty is **just tacked on by addition**. Designers of ridge *decided* "let's also punish big β by adding their squares."

Pick λ = 1. Evaluate `L_ridge` at several candidate β values:

| β₁   | β₂   | RSS  | Penalty β₁²+β₂²    | L_ridge = RSS + λ·penalty |
|------|------|------|--------------------|----------------------------|
| −1   | 3    | 0    | 1 + 9 = 10         | 0 + 10 = **10**            |
| 0    | 2.4  | 0.20 | 0 + 5.76 = 5.76    | 0.20 + 5.76 = **5.96**     |
| 0.82 | 1.80 | 0.66 | 0.67 + 3.24 = 3.91 | 0.66 + 3.91 = **4.57**     |

OLS (−1, 3) gives RSS=0 but a huge penalty of 10. Shrinking toward zero gives up some RSS but saves much more on the penalty. The ridge optimum here is **β_ridge ≈ (0.82, 1.80)** — both features kept, both coefficients pulled toward 0, the weird negative β₁ gone.

### Step 3 — LASSO: replace the penalty with L1

LASSO's loss is the same recipe but uses **absolute values** instead of squares:

```
L_lasso(β₁, β₂) = RSS(β₁, β₂)  +  λ · (|β₁| + |β₂|)
```

Pick λ = 1:

| β₁   | β₂   | RSS  | Penalty \|β₁\|+\|β₂\| | L_lasso = RSS + λ·penalty |
|------|------|------|------------------------|----------------------------|
| −1   | 3    | 0    | 1 + 3 = 4              | 0 + 4 = **4.00**           |
| 0.5  | 2    | 0.5  | 0.5 + 2 = 2.5          | 0.5 + 2.5 = **3.00**       |
| 0    | 2.5  | 0.75 | 0 + 2.5 = 2.5          | 0.75 + 2.5 = **3.25**      |
| 0    | 2.4  | 0.20 | 0 + 2.4 = 2.4          | 0.20 + 2.4 = **2.60**      |

The winner is **β_lasso ≈ (0, 2.4)** — feature 1 has been **set exactly to zero**. That's feature selection happening live. The L1 penalty's "corner at zero" makes it cheaper to drop the feature entirely than to keep a small nonzero coefficient.

### Step 4 — Elastic Net: add BOTH penalties

Elastic Net combines them, mixed by a parameter α ∈ [0, 1]:

```
L_enet(β₁, β₂) = RSS(β₁, β₂)
               + λ · [ α·(|β₁| + |β₂|)  +  (1-α)·(β₁² + β₂²) ]
                       ^^^ LASSO part ^^^    ^^^ Ridge part ^^^
```

With α=0.5 and λ=1, the optimum lands between ridge's (0.82, 1.80) and LASSO's (0, 2.4) — still potentially sparse (can zero things out), but with the L2 component keeping correlated features from being arbitrarily dropped.

---

## 3. Why this matters for eGeMAPS — the correlated-feature case

Now imagine `x₁` and `x₂` are **nearly identical** (correlation 0.99) — like `F0_percentile20` and `F0_percentile50` in our eGeMAPS vector, or full-MFCC1 vs voiced-MFCC1V. True generating model: `y = 1·x₁ + 1·x₂`.

| Method | Typical behavior | Result |
|---|---|---|
| **OLS** | Wild coefficients — maybe (50, −48). Sum is ~2 but each is huge. | Unstable, will flip sign across LOSO folds |
| **Ridge** | Splits the weight evenly: (1.0, 1.0) | Stable, keeps both |
| **LASSO** | Arbitrarily picks one: (2.0, 0) or (0, 2.0) — flips across folds! | Unstable feature selection |
| **Elastic Net** | Compromise: (1.0, 1.0) or (0.9, 0.9), rarely splits them apart | Stable *and* can drop truly useless features |

eGeMAPSv02 has many near-duplicate feature pairs. Pure LASSO would give us a *different* "important feature list" every LOSO fold, which is both unstable for prediction and misleading for RQ2 interpretation ("which features actually drive depression signal?"). Ridge is stable but can't tell us which features to drop. Elastic Net is the combination that gives us both properties.

---

## 4. The pattern, restated

Every method minimizes **the same RSS** — the only thing that changes is what penalty term you add next to it:

| Method | Loss function | Result (on correlated features) |
|---|---|---|
| OLS | `RSS` | Unstable, will flip sign across LOSO folds |
| Ridge | `RSS + λ·(β₁² + β₂²)` | Stable, keeps both |
| LASSO | `RSS + λ·(\|β₁\| + \|β₂\|)` | Unstable feature selection |
| Elastic Net | `RSS + λ·[α·(\|β₁\|+\|β₂\|) + (1-α)·(β₁²+β₂²)]` | Stable *and* can drop truly useless features |

The penalty is literally just added on. No hidden derivation. Different penalty shapes (squared vs absolute) give different behavior: ridge shrinks smoothly, LASSO can zero things out, Elastic Net does both.

---

## 5. Implementation plan

In sklearn:

```python
from sklearn.linear_model import ElasticNetCV

model = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],   # α mixing param grid
    alphas=None,                            # let sklearn pick λ grid
    cv=5,                                   # inner CV for hyperparam selection
).fit(X_train, y_train)
```

Nested inside the LOSO outer loop. `l1_ratio=1.0` reduces to pure LASSO, `l1_ratio=0` to pure Ridge — the CV picks the mix for us, so we don't have to commit to one point on the spectrum in advance.

This matches the plan in `CLAUDE.md`: "Run PLS, Elastic Net, and per-task-PCA+ridge in the same LOSO harness and compare."

---

## 6. Honest caveats

- With n=52 and LOSO (n_train=51 per fold), Elastic Net still needs careful hyperparameter tuning. The `l1_ratio` and `alphas` grid search must happen on an *inner* CV inside each LOSO training fold — never on the held-out subject.
- If Elastic Net selects all 88 features with tiny coefficients at the CV-chosen λ, that's functionally ridge; if it selects one per correlated group, that's functionally LASSO. We should report which regime it lands in.
- Pre-pruning via the literature review (see `eGeMAPS_Feature_Pruning_Review.md`) is complementary, not competing: filter to ~50 features based on prior evidence, *then* let Elastic Net do final selection inside the CV loop.
