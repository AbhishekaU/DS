"""
Stage 5 — Evaluation.

The train/test split happened back in Stage 3. This is the first file that
actually looks at X_test's predictions — everything in Stage 4 (cross-
validation) deliberately never touched it, so this one look is still an honest
measure of how the model does on data it never learned from.

Steps:
1. Fit the Stage 4 winner (LogReg + class_weight) on the FULL training set.
2. Predict probabilities (not just Yes/No) on the test set.
3. Look at metrics at the default 0.5 threshold — then show why 0.5 is an
   arbitrary cutoff, not a correct one, for this problem.
4. Sweep thresholds and pick one by actual business cost: a missed churner
   (false negative) costs a customer's yearly revenue; a false alarm (false
   positive) costs a retention offer we didn't need to give.
5. Save the confusion matrix, ROC curve, and cost-vs-threshold plots to
   reports/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
)

from preprocessing import load_data, clean_data, split_data, build_preprocessor

# Business-cost assumptions (illustrative, clearly separate from the metrics
# above them — these are judgment calls a real retention team would set, not
# something derived from the data itself):
#   - Missing a churner (false negative): we lose that customer's revenue for
#     roughly a year -> cost = their MonthlyCharges * 12. Data-driven per
#     customer, not a flat guess.
#   - Wrongly flagging a happy customer (false positive): we spend a retention
#     offer/discount on someone who didn't need it -> flat $50 assumption.
FN_COST_PER_MONTH_MULTIPLIER = 12
FP_FLAT_COST = 50


def evaluate_at_threshold(y_true, y_proba, monthly_charges, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    total_cost = (
        monthly_charges[fn_mask].sum() * FN_COST_PER_MONTH_MULTIPLIER
        + fp_mask.sum() * FP_FLAT_COST
    )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "total_cost_$": total_cost,
    }


if __name__ == "__main__":
    raw = load_data()
    cleaned = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(cleaned)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)  # first real use of the test set

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_t, y_train)

    y_proba = model.predict_proba(X_test_t)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    print("=== Metrics at default threshold (0.5) ===")
    print(classification_report(y_test, y_pred_default, target_names=["No churn", "Churn"]))
    print(f"ROC-AUC (threshold-independent): {roc_auc_score(y_test, y_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred_default)
    print("Confusion matrix [[TN, FP], [FN, TP]]:\n", cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix @ threshold=0.5")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix_default.png", dpi=120)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=120)
    plt.close(fig)

    # --- Threshold sweep, judged by business cost, not just accuracy ---
    print("\n=== Threshold sweep (cost = missed churner's ~annual revenue + $50 per false alarm) ===")
    monthly_charges = X_test["MonthlyCharges"].to_numpy()
    thresholds = np.arange(0.05, 0.96, 0.05)  # full range — don't assume where the cost minimum sits
    rows = [evaluate_at_threshold(y_test.to_numpy(), y_proba, monthly_charges, t) for t in thresholds]
    results = pd.DataFrame(rows)
    results["threshold"] = results["threshold"].round(2)  # np.arange floats aren't exact (0.5 != 0.50)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(results.to_string(index=False))

    best_row = results.loc[results["total_cost_$"].idxmin()]
    print(f"\nLowest-cost threshold: {best_row['threshold']:.2f} "
          f"(total estimated cost ${best_row['total_cost_$']:.0f}, "
          f"vs ${results[results['threshold'] == 0.50]['total_cost_$'].values[0]:.0f} at the default 0.5)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results["threshold"], results["total_cost_$"], marker="o")
    ax.axvline(best_row["threshold"], color="green", linestyle="--", label=f"lowest cost @ {best_row['threshold']:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", label="default 0.5")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Estimated total cost ($)")
    ax.set_title("Business cost vs. decision threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/cost_vs_threshold.png", dpi=120)
    plt.close(fig)

    print("\nSaved: reports/confusion_matrix_default.png, reports/roc_curve.png, reports/cost_vs_threshold.png")
