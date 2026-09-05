"""
Stage 2 — Cleaning & feature engineering.

Everything here traces back to a finding from `01_eda.ipynb`:

- `TotalCharges` is stored as text and hides 11 blank rows, all `tenure == 0`
  (brand-new customers, no invoice yet, none churned) -> we fill those with 0
  rather than the column mean, since "average spend" has no meaning for a
  customer who hasn't been billed.
- `Contract` has a real order (Month-to-month < One year < Two year) and was the
  single strongest churn signal in the EDA -> encoded ordinally, not one-hot, so
  a linear model can use "more committed = less likely to churn" directly.
- The other service columns (`InternetService`, `OnlineSecurity`, ...) have no
  natural order -> one-hot encoded.
- Plain Yes/No and gender columns carry no information loss when mapped to 0/1
  directly (it's a fixed, data-independent lookup) -> mapped once, upfront, on
  the *whole* dataframe. This is safe and is NOT the same thing as fitting an
  encoder/scaler, which must only ever be fit on the training split (Stage 3) or
  it leaks information from the test set into training.
- `tenure_group` is a new bucketed feature (0-12, 13-24, 25-48, 49-60, 61-72
  months) — gives tree models an explicit "new customer" signal instead of
  relying on them to discover the same cutoff in raw `tenure`.

Stage 3 — train/test split + imbalance handling — lives here too:

- **Stratified split**: `train_test_split(..., stratify=y)` keeps the ~73.5/26.5
  churn ratio the same in both the train and test sets. Without it, a plain
  random split could easily hand the test set a noticeably different churn rate
  by chance, making the evaluation numbers less trustworthy.
- **The `ColumnTransformer` is fit on `X_train` only**, then just `.transform()`ed
  on `X_test` — this is the real usage the Stage 2 smoke test was deliberately
  NOT doing.
- **Two ways to handle the 73.5/26.5 imbalance, compared side by side:**
  `class_weight="balanced"` (tell the model to penalize missing a churner more
  heavily, without touching the data) vs. `SMOTE` (generate synthetic churner
  examples so the training set is 50/50). Both are applied to `X_train` only —
  never to `X_test`, or you'd be evaluating against a test set that no longer
  reflects reality.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/Telco-Customer-Churn.csv"

# Columns with a genuine Yes/No (or Female/Male) meaning -> safe to map to 0/1
# directly; this is a fixed lookup, not something "learned" from data.
BINARY_MAP_COLUMNS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "Churn": {"No": 0, "Yes": 1},
}

# Contract has a real order -> ordinal, not one-hot.
ORDINAL_COLUMNS = ["Contract"]
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]

# No natural order -> one-hot. Includes the engineered tenure_group bucket.
NOMINAL_COLUMNS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaymentMethod", "tenure_group",
]

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues and engineer new features. Deterministic —
    safe to run on the full dataset before splitting.

    Also used at inference time (Stage 8) on a single new customer, who has no
    `customerID` and no `Churn` yet (that's what we're predicting) — both are
    handled as optional so this one function serves training and live
    prediction alike, instead of duplicating cleaning logic in the app."""
    df = df.copy()

    # customerID is a unique identifier, not a predictive feature. Optional:
    # absent for a live prediction row.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges: text -> numeric; the 11 unparseable rows are tenure==0
    # customers who haven't been billed yet, so 0 is the correct fill, not the
    # column mean (which would invent spend that never happened).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Engineered feature: bucket tenure into named groups.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    ).astype(str)

    # Deterministic binary mappings (not "fit" on data — safe pre-split).
    # "Churn" is optional: absent for a live prediction row (that's the
    # unknown we're predicting), present for every training row.
    for col, mapping in BINARY_MAP_COLUMNS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Returns an UNFIT ColumnTransformer. Fitting must happen only on the
    training split (Stage 3) to avoid leaking test-set statistics (e.g. the mean
    used by StandardScaler, or which categories OneHotEncoder learns) into
    evaluation."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("ord", OrdinalEncoder(categories=[CONTRACT_ORDER]), ORDINAL_COLUMNS),
            ("nom", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL_COLUMNS),
        ],
        remainder="passthrough",  # already-binary 0/1 columns pass through untouched
    )


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified split so both sets keep the same ~73.5/26.5 churn ratio as the
    full dataset."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def apply_smote(X_train_transformed, y_train, random_state: int = 42):
    """Synthetic Minority Oversampling: for each churner, look at its nearest
    churner neighbors (in the transformed feature space) and generate new,
    synthetic churner rows along the line between them — until churners and
    non-churners are balanced. ONLY ever call this on X_train — applying it to
    X_test would evaluate the model against fake data and invalidate the
    metrics."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train_transformed, y_train)


if __name__ == "__main__":
    raw = load_data()
    cleaned = clean_data(raw)

    print("Shape after cleaning:", cleaned.shape)
    print("\nAny remaining nulls?\n", cleaned.isna().sum().sum())
    print("\ndtypes:\n", cleaned.dtypes)
    print("\ntenure_group counts:\n", cleaned["tenure_group"].value_counts())

    # --- Stage 3: real split + imbalance handling ---
    X_train, X_test, y_train, y_test = split_data(cleaned)
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("Train churn ratio:\n", y_train.value_counts(normalize=True).round(3))
    print("Test churn ratio:\n", y_test.value_counts(normalize=True).round(3))

    # Fit the preprocessor on X_train ONLY, then transform both.
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    print(f"\nTransformed train shape: {X_train_t.shape}, test shape: {X_test_t.shape}")
    print("Feature names:", list(preprocessor.get_feature_names_out()))

    # Option A: class_weight="balanced" — no data change, just note the weights
    # a model would use (computed the same way sklearn does internally).
    n_no, n_yes = y_train.value_counts()[0], y_train.value_counts()[1]
    print(f"\n[class_weight option] would weight churn=1 ~{n_no / n_yes:.2f}x "
          f"more heavily than churn=0 (no synthetic rows created)")

    # Option B: SMOTE — actually resample X_train (transformed) to 50/50.
    X_train_smote, y_train_smote = apply_smote(X_train_t, y_train)
    print(f"\n[SMOTE option] train shape before: {X_train_t.shape}, after: {X_train_smote.shape}")
    print("[SMOTE option] class balance after:\n", y_train_smote.value_counts(normalize=True).round(3))
