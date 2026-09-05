# Customer Churn Prediction — End-to-End ML Project

A classic **supervised classification** project built end-to-end: raw CSV → EDA →
cleaning/feature engineering → model training & comparison → evaluation →
explainability → a **Streamlit** app that scores a customer's churn risk live.

This is a learning project, deliberately built **one stage at a time** so every ML
concept (imbalance handling, encoding, cross-validation, metric choice,
explainability) is understood before moving to the next — same approach as
[`Event-Planner-AI`](../Event-Planner-AI).

> Companion file: [`learning.md`](./learning.md) — a running log of every ML concept
> covered, stage by stage, as we build.

---

## 1. Problem Statement

A telecom company wants to know **which customers are likely to cancel their
subscription (churn)** so retention teams can act before they leave. Given a
customer's account and usage attributes, predict the probability they churn.

- **Task type:** Binary classification (`Churn`: Yes/No)
- **Why this dataset is a good teacher:** real-world messiness (mixed types, a
  numeric column stored as text with blanks, class imbalance ~73/27), and a business
  outcome that makes every metric decision meaningful (is a false negative worse than
  a false positive here? — yes, so recall matters).

## 2. Dataset

**IBM Telco Customer Churn** — 7,043 customers, 20 features + target.

| Group | Columns |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Charges | `MonthlyCharges`, `TotalCharges` |
| Target | `Churn` (Yes/No) |

Stored at [`data/Telco-Customer-Churn.csv`](./data/Telco-Customer-Churn.csv).

## 3. Pipeline (built stage by stage)

We only move to the next stage once the current one is understood — no code lands
until the concept behind it is clear.

| # | Stage | File | Status |
|---|---|---|---|
| 1 | Exploratory Data Analysis | `01_eda.ipynb` | ✅ done |
| 2 | Cleaning & feature engineering | `preprocessing.py` | ✅ done |
| 3 | Train/test split + imbalance handling | `preprocessing.py` | ✅ done |
| 4 | Model training & comparison (LogReg, RandomForest, XGBoost) | `train.py` | ✅ done |
| 5 | Evaluation (precision/recall/F1/ROC-AUC, confusion matrix) | `evaluate.py` | ✅ done |
| 6 | Explainability (SHAP) | `explain.py` | ✅ done |
| 7 | Save best model (`joblib`) | `models/` | ✅ done |
| 8 | Streamlit app — live churn-risk scorer + Gemini summary | `app.py` | ✅ done |

### Stage 1 — Exploratory Data Analysis (`01_eda.ipynb`)
What we do: load the CSV, check shape/dtypes/nulls, look at the target balance
(`Churn` Yes/No split), plot how churn varies against tenure, contract type, monthly
charges, and internet service; spot the `TotalCharges` column being stored as text
with 11 blank rows.
What you learn: reading a `describe()`/`info()` output like a diagnosis, spotting a
data-quality bug before it silently breaks a model, and using cross-tabs/plots to
form hypotheses about *which* features will matter — before any model confirms it.

### Stage 2 — Cleaning & feature engineering (`preprocessing.py`)
What we do: coerce `TotalCharges` to numeric and decide what to do with the blanks
(they're all `tenure == 0`, i.e. brand-new customers — drop vs. impute is a real
decision we'll reason through); encode categoricals (one-hot for nominal columns
like `PaymentMethod`, ordinal/binary for `Contract`, `Yes`/`No` columns); optionally
engineer a `tenure_group` bucket feature.
What you learn: why models need numeric input, one-hot vs. ordinal encoding and when
each applies, and why "just drop missing rows" is a decision with consequences, not
a default.

### Stage 3 — Train/test split + imbalance handling (`preprocessing.py`)
What we do: stratified train/test split (so the ~27% churn rate holds in both sets),
then compare two ways of handling the imbalance: `class_weight='balanced'` vs.
SMOTE oversampling on the training fold only.
What you learn: why splitting *before* balancing matters (to avoid leaking synthetic
test-like data into evaluation), what stratification protects against, and the
trade-offs between reweighting vs. resampling.

### Stage 4 — Model training & comparison (`train.py`)
What we do: train Logistic Regression (baseline), Random Forest, and XGBoost on the
same split; use cross-validation to sanity-check each before touching the test set.
What you learn: why you always beat a simple baseline first, how tree ensembles
capture non-linear feature interactions a linear model can't, and what
cross-validation actually protects you from (overfitting to one lucky split).

### Stage 5 — Evaluation (`evaluate.py`)
What we do: compare models on precision, recall, F1, and ROC-AUC — not accuracy —
plus a confusion matrix; decide a probability threshold based on what a false
negative costs the business (a churner missed) vs. a false positive (a retention
offer wasted).
What you learn: why accuracy lies on imbalanced data, what each metric actually
answers, and that "best model" depends on which mistake is more expensive here.

### Stage 6 — Explainability (`explain.py`)
What we do: run SHAP on the winning model to see which features push a *specific*
customer's prediction up or down, and which features matter most overall.
What you learn: the difference between global importance ("tenure matters") and
local explanation ("this customer's short tenure + month-to-month contract is why
they're flagged"), and why that distinction matters for a retention team acting on
the model.

### Stage 7 — Save the best model (`models/`)
What we do: serialize the trained pipeline (preprocessing + model together) with
`joblib` so the Streamlit app doesn't need to retrain.
What you learn: why you persist the *whole* pipeline, not just the model — otherwise
inference-time preprocessing can silently drift from training-time preprocessing.

### Stage 8 — Streamlit app (`app.py`)
What we do: a form for a customer's attributes → load the saved pipeline → show
churn probability, risk tier, and a chart against the Stage 5 threshold → a SHAP
waterfall for that specific prediction → **Gemini (via `langchain_google_genai`,
same pattern as `Event-Planner-AI`) turns the SHAP breakdown into a short,
jargon-free explanation and 2-3 concrete retention actions**, for a customer
support/sales reader who's never heard of SHAP or logistic regression.
What you learn: turning a notebook result into something a non-technical person
could actually use — same deployment muscle as `Event-Planner-AI`, applied to a
predictive model instead of an agent; and the pattern of pairing a deterministic
model (does the math, stays consistent) with an LLM (does the translation to
plain English) rather than asking either one to do the other's job.

## 4. Why these choices

- **Baseline first (Logistic Regression):** cheap, interpretable, gives a floor to
  beat before reaching for ensembles.
- **Random Forest / XGBoost:** handle mixed categorical/numeric features and
  non-linear interactions well without heavy preprocessing.
- **Class imbalance:** ~27% churn rate — accuracy alone would be misleading (a
  model predicting "No" always scores ~73%). We'll use `class_weight='balanced'`
  and/or SMOTE, and judge models on **recall/F1/ROC-AUC**, not accuracy.
- **SHAP over plain feature_importances_:** shows *direction* of effect per
  feature per prediction, not just magnitude — the difference between "tenure
  matters" and "short tenure increases churn risk."
- **Streamlit deployment:** consistent with `Event-Planner-AI`, and turns the
  model into something demoable, not just a notebook metric.

## 5. Setup

```bash
cd Churn-Prediction-ML
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file with a Gemini API key (used by Stage 8's plain-English
summary — the rest of the pipeline doesn't need it):

```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.6-flash   # optional, this is the default
```

## 6. Run

```bash
python preprocessing.py   # Stage 2/3 sanity check
python train.py           # Stage 4 model comparison
python evaluate.py        # Stage 5 test-set evaluation + reports/
python explain.py         # Stage 6 SHAP + reports/
python save_model.py      # Stage 7 — writes models/churn_pipeline.joblib
streamlit run app.py      # Stage 8 — needs models/churn_pipeline.joblib to exist
```
