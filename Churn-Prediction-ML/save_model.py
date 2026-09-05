"""
Stage 7 — Save the trained pipeline.

Fits the Stage 4 winner (LogReg + class_weight) on the FULL training set, one
final time, and persists it together with the fitted ColumnTransformer AND the
Stage 5 cost-optimal decision threshold (0.10) — all three bundled into one
file with `joblib`.

Why bundle all three, not just the model:
- Saving the model alone would leave Stage 8 (the Streamlit app) to reconstruct
  the preprocessing by hand — exactly the "did I scale/encode it the same way
  as training?" risk flagged back in Stage 2. Saving the FITTED preprocessor
  guarantees the app applies the identical transformation, using the identical
  learned mean/std/categories from training, every time.
- Saving the threshold alongside it means Stage 8 doesn't silently fall back to
  Python's default 0.5 (which Stage 5 showed costs ~46% more) — the actual
  reasoned decision travels with the model, not as a magic number pasted into
  app.py separately where it could drift out of sync.

Note: `clean_data()` (customerID drop, TotalCharges fix, tenure_group bucket,
binary mapping) is intentionally NOT part of the saved object — it's a cheap,
deterministic, stateless function (no "fit" step, nothing learned from data),
so Stage 8 just imports and calls it directly from preprocessing.py, exactly
like every other stage has.
"""

import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from preprocessing import load_data, clean_data, split_data, build_preprocessor

MODEL_PATH = "models/churn_pipeline.joblib"
CHOSEN_THRESHOLD = 0.10  # from Stage 5's cost-vs-threshold sweep

if __name__ == "__main__":
    raw = load_data()
    cleaned = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(cleaned)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    # Sanity check: this should reproduce Stage 5's test ROC-AUC exactly, since
    # it's the same model/data/random_state — just now saved as one object
    # instead of two fit separately. A mismatch here would mean something
    # about the bundling changed behavior.
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"Sanity check — test ROC-AUC via saved pipeline: {test_auc:.4f} "
          f"(Stage 5 reported 0.8422 — should match)")

    joblib.dump({"pipeline": pipeline, "threshold": CHOSEN_THRESHOLD}, MODEL_PATH)
    print(f"Saved: {MODEL_PATH} (pipeline + threshold={CHOSEN_THRESHOLD})")

    # Reload check: prove the file on disk actually works standalone, the way
    # Stage 8 will use it — a fresh load, no training-time objects in memory.
    reloaded = joblib.load(MODEL_PATH)
    reloaded_auc = roc_auc_score(y_test, reloaded["pipeline"].predict_proba(X_test)[:, 1])
    assert abs(reloaded_auc - test_auc) < 1e-9, "Reloaded model doesn't match the one just saved!"
    print(f"Reload check passed — reloaded pipeline gives identical ROC-AUC: {reloaded_auc:.4f}")
