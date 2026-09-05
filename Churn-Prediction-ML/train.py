"""
Stage 4 — Model training & comparison.

Trains 3 models (Logistic Regression, Random Forest, XGBoost), each under both
imbalance strategies from Stage 3 (class_weight vs SMOTE) — 6 combinations total
— and cross-validates all of them on the TRAINING set only, before Stage 5 ever
touches the test set. This is the "sanity-check before you spend your one shot
at the test set" step from the README.

Key subtlety this stage teaches: SMOTE must be re-applied fresh inside EACH
cross-validation fold's training portion, not resampled once upfront and then
cross-validated on top of that. If you resampled once upfront, a synthetic row's
"parent" real rows could end up split across the train/validation portions of a
fold — meaning the validation fold would partly evaluate against data that
influenced the synthetic rows it's judging. imblearn's `Pipeline` handles this
correctly: it's SMOTE-aware and re-fits SMOTE on only each fold's training
portion automatically.

Baseline first, always: Logistic Regression is included specifically so the
ensembles (Random Forest, XGBoost) have something simple to beat before we trust
their added complexity.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from preprocessing import load_data, clean_data, split_data, build_preprocessor

SCORING = ["roc_auc", "f1", "recall", "precision"]


def build_candidates(scale_pos_weight: float) -> dict:
    """Returns {name: estimator_or_pipeline} for all 6 (model x imbalance
    strategy) combinations. `scale_pos_weight` is XGBoost's equivalent of
    class_weight="balanced" — it doesn't accept that string, only a numeric
    ratio (computed from the training data in __main__ below)."""

    # --- Option A: class_weight="balanced" (no data resampling) ---
    logreg_cw = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    rf_cw = RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=42)
    xgb_cw = XGBClassifier(
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42
    )

    # --- Option B: SMOTE (imblearn Pipeline re-applies it fresh per CV fold) ---
    logreg_smote = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    rf_smote = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ])
    xgb_smote = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf", XGBClassifier(eval_metric="logloss", random_state=42)),
    ])

    return {
        "LogReg + class_weight": logreg_cw,
        "LogReg + SMOTE": logreg_smote,
        "RandomForest + class_weight": rf_cw,
        "RandomForest + SMOTE": rf_smote,
        "XGBoost + class_weight": xgb_cw,
        "XGBoost + SMOTE": xgb_smote,
    }


def cross_validate_candidates(candidates: dict, X_train_t, y_train) -> pd.DataFrame:
    """5-fold stratified CV on the training set only. Returns a results table
    sorted by mean ROC-AUC (chosen as the primary ranking metric since it's
    threshold-independent — Stage 5 picks the actual decision threshold)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in candidates.items():
        scores = cross_validate(model, X_train_t, y_train, cv=cv, scoring=SCORING)
        rows.append({
            "model": name,
            "roc_auc": scores["test_roc_auc"].mean(),
            "f1": scores["test_f1"].mean(),
            "recall": scores["test_recall"].mean(),
            "precision": scores["test_precision"].mean(),
        })
    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return results


if __name__ == "__main__":
    raw = load_data()
    cleaned = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(cleaned)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    # X_test is intentionally untouched in this file — Stage 5's job.

    n_no, n_yes = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = n_no / n_yes

    candidates = build_candidates(scale_pos_weight)
    results = cross_validate_candidates(candidates, X_train_t, y_train)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("5-fold cross-validation results on TRAINING data (test set untouched):\n")
    print(results.to_string(index=False))
    print(f"\nBest by ROC-AUC: {results.iloc[0]['model']}")
