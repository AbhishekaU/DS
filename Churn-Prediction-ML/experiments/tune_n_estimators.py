"""
One-off experiment (not part of the pipeline stages): does n_estimators=300 in
train.py actually earn its keep, or would fewer trees do just as well?

Uses Random Forest's built-in OOB (out-of-bag) score: since each tree is trained
on a random ~63% bootstrap sample of the training data, the ~37% it never saw
can score that tree for free — no separate cross-validation split needed. We
watch how the OOB score changes as n_estimators grows and look for the point
where adding more trees stops helping (the "elbow").

Caveat we're deliberately testing: the default OOB score is accuracy-based,
which Stage 1/5 already flagged as misleading on our ~73.5/26.5 imbalanced data.
So we track OOB accuracy (fast, free) AND cross-validated ROC-AUC (the metric we
actually care about) side by side, to see whether the free/fast OOB signal
agrees with the real metric on WHERE the plateau is — even if OOB's raw number
isn't the one we'd trust for a final score.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import load_data, clean_data, split_data, build_preprocessor

N_VALUES = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000]

if __name__ == "__main__":
    raw = load_data()
    cleaned = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(cleaned)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for n in N_VALUES:
        rf = RandomForestClassifier(
            n_estimators=n, class_weight="balanced", oob_score=True,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train_t, y_train)
        oob = rf.oob_score_

        cv_auc = cross_val_score(
            RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=42, n_jobs=-1),
            X_train_t, y_train, cv=cv, scoring="roc_auc",
        ).mean()

        rows.append({"n_estimators": n, "oob_accuracy": oob, "cv_roc_auc": cv_auc})
        print(f"n_estimators={n:>5}  oob_accuracy={oob:.4f}  cv_roc_auc={cv_auc:.4f}")

    results = pd.DataFrame(rows)
    results["oob_gain"] = results["oob_accuracy"].diff()
    results["auc_gain"] = results["cv_roc_auc"].diff()

    print("\nStep-to-step gains (how much each jump in n_estimators helped):\n")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results.to_string(index=False))
