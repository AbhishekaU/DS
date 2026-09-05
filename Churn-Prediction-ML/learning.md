# Learning Log — Customer Churn Prediction

Running notes on every ML concept covered, stage by stage, as we build
`Churn-Prediction-ML`. See [`README.md`](./README.md) for the overall plan.

---

## Stage 0 — Project setup

- Picked **classification** (churn: Yes/No) over regression/time-series/CV to get
  practice with the classic supervised-learning pipeline: EDA → cleaning → encoding →
  imbalance handling → model comparison → metric-driven evaluation → explainability →
  deployment.
- Dataset: IBM Telco Customer Churn (7,043 rows, 20 features + target), pulled from a
  public GitHub mirror into `data/Telco-Customer-Churn.csv`.

---

## Stage 1 — Exploratory Data Analysis (`01_eda.ipynb`)

Built and **executed end-to-end** (real outputs baked into the notebook, not
placeholders) against a venv with pandas/matplotlib/seaborn installed.

**Concepts covered:**

- **`info()` doesn't catch everything.** `TotalCharges` showed up as "non-null
  object" even though 11 rows hold a blank string (`" "`), not `NaN`. A column can
  pass the null-check and still be unusable as numeric data — you have to actually
  try `pd.to_numeric(..., errors="coerce")` and see what breaks.
- **Root-causing a data bug instead of papering over it.** All 11 broken
  `TotalCharges` rows turned out to have `tenure == 0` — brand-new customers with no
  invoice yet. That's *not* missing data in the usual sense, so "impute with the
  mean" would be the wrong fix. This is deferred to Stage 2 as a decision (0 vs.
  drop), not solved blindly here.
- **Why accuracy alone is a trap.** Churn is ~73.5% No / 26.5% Yes. A model that
  always predicts "No" scores ~73.5% accuracy while catching zero churners. This is
  *why* Stage 5 will use recall/F1/ROC-AUC instead of accuracy, and why Stage 3
  needs imbalance handling at all.
- **Group-wise rates vs. correlation coefficients tell different stories.**
  `Contract` and `InternetService` are categorical, so a single Pearson correlation
  number (which only applies to the numeric columns) would've hidden their real
  signal. Group-wise churn rates showed month-to-month customers churn ~42.7% vs.
  ~2.8% for two-year contracts — the strongest pattern in the whole dataset — and
  that would never show up in a numeric-only correlation heatmap.
- **Collinearity spotted before modeling.** `tenure` (−0.35 corr with churn) and
  `TotalCharges` (−0.20) move together because `TotalCharges` is partly just
  "tenure × monthly rate" accumulated — flagged now so Stage 4 doesn't treat them
  as two independent signals.

**Actual numbers found** (for reference going into later stages):
tenure corr = −0.35, TotalCharges corr = −0.20, SeniorCitizen corr = +0.15,
MonthlyCharges corr = +0.19; churn rate by contract: month-to-month 42.7%, one-year
11.3%, two-year 2.8%; churn rate by internet service: fiber 41.9%, DSL 19.0%, none
7.4%.

---

## Stage 2 — Cleaning & feature engineering (`preprocessing.py`)

**Concepts covered:**

- **Deterministic transforms vs. fit-on-train-only transforms.** Mapping
  `Yes`/`No` → `1`/`0` is a fixed lookup — it doesn't "learn" anything from the
  data, so it's safe to apply to the whole dataframe before splitting. Contrast
  that with `StandardScaler` (learns a mean/std) or `OneHotEncoder` (learns which
  categories exist) — those must only ever be fit on the *training* split, or
  statistics from the test set leak into training and inflate your evaluation
  metrics. This is why `build_preprocessor()` returns an **unfit**
  `ColumnTransformer` — Stage 3 fits it on `X_train` only.
- **Ordinal vs. one-hot encoding, applied for real.** `Contract` has a genuine
  order (Month-to-month < One year < Two year) matching what EDA found (churn
  rate drops as commitment increases) — encoded with `OrdinalEncoder` so a linear
  model can use "more committed → less likely to churn" as a single coefficient.
  Columns like `InternetService` or `PaymentMethod` have no such order, so
  one-hot is correct there — an ordinal encoding would invent a fake ranking
  (e.g. "Mailed check < Electronic check") that doesn't exist.
- **Fixing the `TotalCharges` blanks the way EDA justified**, not the generic
  way: filled with `0`, not the column mean, because Stage 1 showed all 11 blanks
  are `tenure == 0` customers with genuinely nothing billed yet — the mean would
  have invented spend that never happened.
- **Feature engineering as a modeling aid, not just cosmetics.** `tenure_group`
  buckets tenure into 5 named ranges. Tree models *can* discover a cutoff like
  "tenure < 12" on their own by splitting repeatedly, but handing them the bucket
  directly is cheaper and makes the "new customer" signal explicit and
  inspectable later in SHAP (Stage 6).
- **A `ColumnTransformer` bundles all of this into one object** that Stage 3 fits
  once on the training data, Stage 4 feeds into every model, and Stage 7 saves
  *alongside* the trained model (not separately) — so Stage 8's Streamlit app
  can't accidentally apply different preprocessing at inference time than what
  the model was trained on.

**Result:** 7,043 rows, 0 remaining nulls, 20 cleaned input columns → 33 features
after encoding (verified by running `preprocessing.py` directly — output feature
names print in full: 3 scaled numeric, 1 ordinal, ~22 one-hot, 7 passthrough
binary).

---

## Stage 3 — Train/test split + imbalance handling (`preprocessing.py`)

**Concepts covered:**

- **Stratified split.** `train_test_split(..., stratify=y)` forces both the
  train and test sets to keep the same 73.5/26.5 churn ratio as the full
  dataset. Verified: train = 73.5%/26.5%, test = 73.5%/26.5% — an exact match,
  which a plain random split would only hit by luck.
- **Fit on train, transform on test — for real this time.** Stage 2's `__main__`
  deliberately fit the `ColumnTransformer` on the *full* dataset as a labeled
  smoke test. Stage 3 does it correctly: `preprocessor.fit_transform(X_train)`
  learns the scaler's mean/std and the encoder's categories from training data
  only, then `preprocessor.transform(X_test)` reuses those exact learned values
  on the test set without re-learning anything from it.
- **Two competing fixes for the 73.5/26.5 imbalance, actually compared:**
  - **`class_weight="balanced"`** — doesn't touch the data at all; tells the
    model's loss function to penalize a missed churner (false negative) about
    **2.77x** more heavily than a false positive. Cheap, no synthetic data risk.
  - **`SMOTE`** — generates synthetic churner rows by interpolating between real
    churners' nearest neighbors in feature space, until the classes are 50/50.
    Verified: training set grew from 5,634 → 8,278 rows, landing at an exact
    50.0%/50.0% split.
  - **Both are applied to `X_train` only.** Neither ever touches `X_test` —
    doing so would evaluate the model against a test set that's been reweighted
    or partly synthetic, making the reported metrics meaningless as a measure of
    real-world performance. Which of the two "wins" gets decided empirically in
    Stage 4 by comparing model performance under each.

**Result:** train shape (5,634, 33), test shape (1,409, 33), both preserving the
73.5/26.5 churn ratio; SMOTE-balanced training set (8,278, 33) available as an
alternative to try alongside `class_weight="balanced"` in Stage 4.

---

## Stage 4 — Model training & comparison (`train.py`)

**Concepts covered:**

- **Cross-validation before touching the test set.** `StratifiedKFold` (5-fold)
  repeatedly splits the *training* data into 5 mini train/validation slices,
  trains on 4/5 and scores on the held-out 1/5, then averages across all 5. This
  gives a much more reliable performance estimate than one lucky/unlucky single
  split — and critically, `X_test` is never touched in this file at all.
- **SMOTE must be re-applied inside each CV fold, not once upfront.** If you
  resampled the whole training set with SMOTE first and *then* cross-validated
  on top of that, a synthetic row's "parent" real rows could land in a
  different fold than the synthetic row itself — so a validation fold could
  partly be scored against data that leaked into the synthetic rows it's
  judging. `imblearn.pipeline.Pipeline` avoids this automatically: it re-fits
  SMOTE fresh on only each fold's training portion.
- **ROC-AUC as the ranking metric, on purpose.** It's threshold-independent
  (measures ranking quality across every possible probability cutoff), so it
  doesn't bake in a specific "yes/no" decision boundary — that choice is
  deliberately deferred to Stage 5, where the *cost* of a false negative vs.
  false positive actually gets weighed.

**Actual result — genuinely surprising, and a real lesson, not a foregone
conclusion:**

| Model | ROC-AUC | F1 | Recall | Precision |
|---|---|---|---|---|
| **LogReg + class_weight** | **0.8464** | 0.6273 | 0.7960 | 0.5179 |
| LogReg + SMOTE | 0.8449 | 0.6278 | 0.7860 | 0.5228 |
| RandomForest + class_weight | 0.8276 | 0.6112 | 0.6508 | 0.5762 |
| RandomForest + SMOTE | 0.8242 | 0.5842 | 0.5692 | 0.6005 |
| XGBoost + class_weight | 0.8211 | 0.5996 | 0.6589 | 0.5504 |
| XGBoost + SMOTE | 0.8198 | 0.5779 | 0.5612 | 0.5956 |

**The "baseline" won.** Logistic Regression beat both ensembles on ROC-AUC. Why
this makes sense rather than being a fluke: the EDA findings (Stage 1) were
mostly *monotonic* relationships — churn risk rises steadily as tenure drops,
as contract commitment drops, as monthly charges rise. That's exactly the shape
a linear model is built to capture directly. Random Forest and XGBoost earn
their keep on *non-linear interactions* (e.g. "high charges matter a lot, but
only for month-to-month customers") — and on ~5,600 training rows with mostly
one-directional signals, there may not be enough complex interaction structure
for the extra model capacity to pay off; it mostly adds room to overfit instead.
This is precisely why the README insisted on a baseline first — "beating the
baseline" is a real result here, not a formality, and it could easily have gone
the other way on a different dataset.

**class_weight edged out SMOTE for every model.** Consistent with reasoning,
not just this run: class_weight changes *how much the loss function cares*
about mistakes, using only real data; SMOTE changes *what data exists*,
introducing synthetic points that are only as good as the interpolation
assumption between real neighbors holds up. Cheaper and safer won here.

**Carried into Stage 5:** `LogReg + class_weight` is the model to fit on the
full training set and finally evaluate against the untouched test set.

### Side experiment — was `n_estimators=300` actually justified?

(`experiments/tune_n_estimators.py`, not part of the numbered pipeline —
answers "is there math for choosing this?" empirically instead of guessing.)

Trained Random Forest at `n_estimators` = 10 up to 1000, tracking two things at
each value: the free **OOB accuracy** (each tree scored on the ~37% of training
rows it never saw, no extra CV needed) and the real **cross-validated ROC-AUC**
(the metric that actually matters here) — to see whether the free, fast OOB
signal points to the same "enough trees" cutoff as the expensive, trustworthy
one.

| n_estimators | OOB accuracy | CV ROC-AUC | ROC-AUC gain vs. previous |
|---|---|---|---|
| 10 | 0.7123 | 0.8014 | — |
| 50 | 0.7591 | 0.8235 | +0.0057 (from 25) |
| 100 | 0.7717 | 0.8264 | +0.0005 |
| **200** | 0.7755 | **0.8276** | +0.0007 |
| **300** | 0.7748 | **0.8276** | **+0.0000** |
| 500 | 0.7792 | 0.8277 | +0.0002 |
| 1000 | 0.7794 | 0.8280 | +0.0002 |

**Reading the elbow:** both curves climb steeply up to ~100-150 trees, then
flatten hard — ROC-AUC gains per step drop to ~0.0005 or less past 150, and
going all the way to 1000 trees (10x the compute of 300) buys only +0.0004 ROC-AUC
over the 300-tree version. **300 sits right at the plateau** — already past the
point of real returns, not short of it. Fewer trees (say 150) would have scored
essentially identically for cheaper; more trees (1000) cost far more compute for
a rounding-error gain.

**Confirms the general method from the earlier discussion:** there's no formula
that outputs "300" directly — you find the elbow by testing a range and reading
where the curve flattens. What this run adds: OOB accuracy (fast, free, but
built on a flawed metric for imbalanced data per Stage 1) still pointed to
*roughly* the same elbow as the properly-imbalance-aware CV ROC-AUC — useful to
know OOB is a decent quick proxy even when its absolute number shouldn't be
trusted on its own.

---

## Stage 5 — Evaluation (`evaluate.py`)

**First real look at the test set in the whole project.** Every prior stage
(Stage 4's cross-validation included) deliberately never touched `X_test` — this
is the one honest measurement of how the Stage 4 winner (`LogReg +
class_weight`, fit on the full training set) does on data it never learned from.

**Concepts covered:**

- **A confusion matrix names the two ways to be wrong, separately, because they
  aren't equally bad.** At the default threshold: `TP=296, FP=298, FN=78,
  TN=737`. A false negative (78 churners we told the business "they're fine")
  is a much costlier mistake than a false positive (298 loyal customers who got
  an unnecessary retention offer) — accuracy treats both as identically bad,
  which is exactly the flaw Stage 1 flagged.
- **Test-set ROC-AUC (0.8422) landed close to Stage 4's cross-validated estimate
  (0.8464)** — a good sign. If test performance had been dramatically worse than
  CV suggested, that would mean the model had somehow overfit *its
  hyperparameters* to the training folds despite CV, or there was a leak
  somewhere upstream worth re-auditing.
- **0.5 is not a correct threshold, it's just Python's default.** It only makes
  sense if a false positive and a false negative cost exactly the same amount —
  never true here. A missed churner loses ~a year of their revenue
  (`MonthlyCharges × 12`); a false alarm just costs a $50 retention offer we
  didn't strictly need to send. Those are wildly different costs, so the
  threshold should reflect that asymmetry, not default to the midpoint.
- **Threshold sweep, ranked by real dollar cost** (not by precision/recall
  alone) — because precision/recall don't know that a false negative is ~150x
  more expensive here than a false positive; total cost does:

  | Threshold | Recall | Precision | Est. cost |
  |---|---|---|---|
  | 0.10 | 0.99 | 0.34 | **$38,419 (lowest)** |
  | 0.20 | 0.96 | 0.39 | $42,948 |
  | **0.50 (default)** | 0.79 | 0.50 | $71,241 |
  | 0.70 | 0.58 | 0.61 | $131,031 |
  | 0.95 | 0.00 | 1.00 | $325,438 |

  **Lowest-cost threshold: 0.10** — a ~46% reduction in estimated cost
  ($38,419 vs. $71,241) versus just accepting the default 0.5. Makes sense once
  you see the cost asymmetry: since missing a churner is so much more expensive
  than a false alarm, the optimal strategy is to flag almost anyone with even a
  modest churn signal (recall 0.99) and accept a lot of low-cost false alarms
  (precision only 0.34) in exchange.
- **A real bug caught mid-stage, worth remembering:** the first threshold sweep
  only covered 0.20–0.70 and found "0.20" as the best — but cost was *still
  falling* at that edge, meaning the true minimum was outside the tested range.
  Widening the sweep to 0.05–0.95 found the actual minimum at 0.10. Lesson:
  when an optimization lands exactly on the boundary of what you searched,
  that's a signal to widen the search, not a signal you found the answer.
- **Honest caveat worth knowing, not "fixed" here:** the threshold was chosen by
  looking at cost *on the same test set* being used to report final numbers.
  Stricter practice would tune the threshold on a separate validation split (or
  via CV on the training data) and reserve the test set for one truly untouched
  final check — otherwise the threshold choice itself is mildly "informed" by
  the data you're using to claim the model works. Kept simple here since it's a
  learning project, but worth knowing the more rigorous version of this step.

**Business framing worth remembering:** the $50 flat false-positive cost and the
"12 months of revenue" false-negative cost are *assumptions* a real retention
team would set (and could easily be wrong) — not something derived from the
data. Changing those assumptions would shift the optimal threshold; the method
(sweep thresholds, price each type of mistake, pick the minimum) is the
reusable part, not the specific numbers.

**Saved for reference:** `reports/confusion_matrix_default.png`,
`reports/roc_curve.png`, `reports/cost_vs_threshold.png`.

---

## Stage 6 — Explainability (`explain.py`)

**Concepts covered:**

- **Global vs. local explanation are different questions.** Global (SHAP bar/
  beeswarm plots) answers "which features matter most, on average, across
  everyone?" Local (SHAP waterfall for one customer) answers "why did THIS
  specific person get flagged?" A feature can dominate the global ranking while
  playing almost no role in one particular person's prediction, and vice versa
  — which is exactly why Stage 5's confusion matrix alone couldn't tell a
  retention rep what to actually say to a flagged customer.
- **SHAP values are additive and start from a baseline, not zero.** Every
  prediction is explained as: `baseline (E[f(X)]) + each feature's push = final
  score`. The baseline (`E[f(X)] = -0.639` here) is "what the model would say
  with no specific information" — every feature then pushes the prediction up
  (red) or down (blue) from there. This is different from `feature_importances_`
  on a tree, which only gives magnitude, never direction or a starting point.
- **A linear model's global SHAP ranking closely tracked Stage 1's EDA
  hypotheses** — not a coincidence, since a linear model's coefficients (which
  SHAP explains) directly reflect the same monotonic patterns EDA found:

  | Rank | Feature | Mean \|SHAP\| |
  |---|---|---|
  | 1 | `tenure` | 0.916 |
  | 2 | `InternetService_Fiber optic` | 0.598 |
  | 3 | `Contract` | 0.589 |
  | 4 | `MonthlyCharges` | 0.382 |
  | 5 | `StreamingMovies_Yes` | 0.205 |

  Confirms `tenure`, `Contract`, and `MonthlyCharges` (the three strongest EDA
  correlations) are also the three strongest model drivers — the model learned
  what the EDA predicted it would.

- **Revisiting Stage 1's open question: is fiber-optic a real driver, or just
  riding on Contract's coattails?** Correlation between the fiber flag and
  Contract commitment: **−0.244** — fiber customers do skew somewhat toward
  month-to-month, but only mildly, not strongly collinear. And critically,
  fiber optic still ranks **#2 globally by SHAP**, computed *with Contract
  already in the model* — SHAP's game-theoretic attribution already accounts
  for what Contract explains before crediting Fiber with the rest. Conclusion:
  fiber optic carries a real, mostly-independent churn signal (plausibly
  price or reliability-related), not just an echo of contract type.
- **Concrete local example** — the highest-risk test customer (predicted churn
  probability **0.9526**, and they really did churn) breaks down as:
  short tenure (**+1.32**, the single biggest push), fiber optic internet
  (**+0.68**), month-to-month contract (**+0.54**), high monthly charges
  (**+0.26** — note: *lower* than tenure/fiber/contract despite MonthlyCharges
  ranking #4 globally, a reminder that global rank ≠ this person's actual
  breakdown), streaming add-ons and electronic check payment adding smaller
  pushes, partially offset by low TotalCharges (**−0.17**, expected since
  they're new) and a lower MonthlyCharges-adjacent value. This is the
  actionable version of the prediction — a retention rep sees *exactly* which
  levers are pushing this specific person toward churn, not just a 95% number.
- **Background sample size matters for SHAP's baseline precision.** The default
  `shap.Explainer` call subsampled the background reference set to 100 rows
  (a speed shortcut) — noted and then explicitly widened to use all 5,634
  training rows (`shap.maskers.Independent(..., max_samples=X_train.shape[0])`)
  since a linear model is cheap enough to explain exactly, so there was no
  reason to accept the approximation.

**Saved for reference:** `reports/shap_global_bar.png`,
`reports/shap_global_beeswarm.png`, `reports/shap_local_highest_risk.png`.

---

## Stage 7 — Save the trained pipeline (`save_model.py`)

**Concepts covered:**

- **Bundle the fitted preprocessor with the model, not just the model.**
  `Pipeline([("preprocessor", ...), ("classifier", ...)])` fit as one object
  means Stage 8 can never accidentally scale/encode a new customer differently
  than training did — the exact learned mean/std/categories travel with the
  file. This is the payoff promised back in Stage 2's "why a ColumnTransformer"
  explanation, finally cashed in.
- **The decision threshold travels with the model too**, saved alongside it
  (`{"pipeline": ..., "threshold": 0.10}`) — so Stage 8 uses the actual
  cost-reasoned threshold from Stage 5, not Python's default 0.5, and that
  choice can't silently drift out of sync between files.
- **`clean_data()` stays OUTSIDE the saved pipeline, deliberately.** It's
  stateless (nothing "learned" from data — just deterministic fixes and
  bucketing), so there's no leakage risk in calling it fresh each time, and
  keeping it separate means Stage 8 reuses the exact same tested function
  rather than a re-implementation baked into a serialized object.
- **Verified, not assumed, that saving didn't change behavior:** re-computed
  test ROC-AUC through the newly-saved-and-reloaded pipeline and got
  **0.8422** — an exact match with Stage 5's number computed a completely
  different way (fit/transform done separately there vs. as one `Pipeline`
  here). A mismatch would have meant the bundling changed something.

**Saved:** `models/churn_pipeline.joblib` (not committed to git — regenerate
with `python save_model.py`).

---

## Stage 8 — Streamlit app + Gemini summary (`app.py`)

**The idea, and why it's not "just" a form:** the model produces a number,
SHAP explains *why* — but neither of those is something a customer support or
sales rep can act on directly without ML background. This stage adds one more
translation step: an LLM turns the SHAP breakdown into plain English and
concrete next steps. Deliberately split by what each tool is good at — the
model and SHAP are deterministic (same input always gives the same number,
auditable, no hallucination risk) and do all the actual math; the LLM never
touches a number, it only explains ones already computed. Asking an LLM to
*predict* churn (skip the model) or asking a model to *explain itself in
prose* (skip the LLM) would both be worse — using each tool outside what it's
reliable at.

**Concepts covered:**

- **Reused the Stage 2 fix, but from the other direction.** `clean_data()`
  originally handled missing `Churn` for training rows that already have it;
  serving one live customer needed the same function to handle a row that
  never has `Churn` OR `customerID` at all. Small guard added
  (`if "customerID" in df.columns`, same for the `Churn` mapping loop) so one
  function serves both training and inference — no duplicated cleaning logic
  to keep in sync.
- **`st.cache_resource` for anything expensive-and-shared**: the loaded
  pipeline, the SHAP explainer (rebuilding it per request would re-run a
  fit-like step against the full 5,634-row background every time), and the
  Gemini client. Cached once per server process, not once per prediction.
- **`st.form` batches the ~18 inputs** so nothing recomputes (predict, SHAP,
  and a Gemini API call — the expensive path) until "Predict churn risk" is
  actually clicked, not on every keystroke.
- **Design choice: `TotalCharges` isn't a form field.** Stage 1/2 already
  established it's mostly `tenure × MonthlyCharges` — asking a user to type a
  number that contradicts their other two answers would just invite bad
  inputs, so it's computed automatically instead.
- **Verified end-to-end with a real Gemini call**, not mocked — a test customer
  (fiber optic, month-to-month, electronic check, $70.35/mo) came back as
  **68.5% churn probability, High risk**, top SHAP drivers matched Stage 6's
  global ranking (fiber optic +0.68, contract +0.54), and Gemini's summary
  referenced those exact factors in plain English with concrete retention
  offers — no ML jargon, under 150 words, as instructed.
- **The LLM prompt explicitly forbids ML jargon** ("Do NOT mention SHAP,
  coefficients, logistic regression...") — worth remembering as a general
  pattern: an LLM asked to "explain this to a non-technical person" will
  still reach for the technical vocabulary sitting right there in its input
  unless told not to.

**Try it:** `streamlit run app.py`, or see it already running at
`http://localhost:8501` if the dev server from this session is still up.
