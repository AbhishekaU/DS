"""
Stage 6 — Explainability (SHAP).

Stage 4/5 told us the model is good (ROC-AUC ~0.84) and Stage 5 told us WHO it
flags. Neither tells us WHY — and "why" is what actually lets a retention team
act, and what lets us sanity-check the model isn't relying on something
spurious. SHAP (SHapley Additive exPlanations) answers "why" two ways:

- GLOBAL: across all customers, which features push predictions up/down the
  most, and in which direction? (shap summary/bar plots)
- LOCAL: for ONE specific customer, exactly which of their attributes pushed
  their individual risk score up or down, and by how much? (shap waterfall plot)

That local/global distinction is the whole point of this stage — a feature can
matter a lot on average while still not being the reason for any one specific
person's prediction, and vice versa.

We also use this stage to revisit an open question from Stage 1's EDA: is
fiber-optic internet really a churn driver on its own, or is it a confound with
Contract type (fiber customers tending to also be month-to-month)?
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from preprocessing import load_data, clean_data, split_data, build_preprocessor

if __name__ == "__main__":
    raw = load_data()
    cleaned = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(cleaned)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # SHAP needs dense arrays; OneHotEncoder output makes this a sparse matrix.
    X_train_dense = np.asarray(X_train_t.todense()) if hasattr(X_train_t, "todense") else np.asarray(X_train_t)
    X_test_dense = np.asarray(X_test_t.todense()) if hasattr(X_test_t, "todense") else np.asarray(X_test_t)
    feature_names = list(preprocessor.get_feature_names_out())

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_dense, y_train)

    # LogisticRegression is linear -> shap.Explainer picks the exact (not
    # approximate) linear explainer automatically when given this model type.
    # Default background sample is capped at 100 rows for speed; a linear
    # model is cheap to explain, so use the full training set for a more
    # faithful "expected prediction" baseline instead of that shortcut.
    background = shap.maskers.Independent(X_train_dense, max_samples=X_train_dense.shape[0])
    explainer = shap.Explainer(model, background, feature_names=feature_names)
    shap_values = explainer(X_test_dense)

    # --- Global: which features matter most, on average, across all customers ---
    mean_abs_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=feature_names)
    top15 = mean_abs_shap.sort_values(ascending=False).head(15)
    print("=== Top 15 features by mean |SHAP value| (global importance) ===")
    print(top15.to_string())

    fig = plt.figure(figsize=(8, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_global_bar.png", dpi=120)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_global_beeswarm.png", dpi=120)
    plt.close(fig)

    # --- Revisit the Stage 1 open question: fiber-optic, real driver or confound? ---
    fiber_idx = feature_names.index("nom__InternetService_Fiber optic")
    contract_idx = feature_names.index("ord__Contract")
    fiber_shap_mean = shap_values.values[:, fiber_idx].mean()
    contract_shap_mean = shap_values.values[:, contract_idx].mean()
    print(f"\nMean SHAP contribution — InternetService_Fiber optic: {fiber_shap_mean:.4f}")
    print(f"Mean SHAP contribution — Contract (ordinal, higher=more committed): {contract_shap_mean:.4f}")
    corr_fiber_contract = np.corrcoef(X_test_dense[:, fiber_idx], X_test_dense[:, contract_idx])[0, 1]
    print(f"Correlation between (scaled/encoded) Fiber flag and Contract: {corr_fiber_contract:.4f}")

    # --- Local: explain one specific high-risk customer ---
    y_proba_test = model.predict_proba(X_test_dense)[:, 1]
    highest_risk_idx = int(np.argmax(y_proba_test))
    print(f"\nHighest-risk test customer: index {highest_risk_idx}, "
          f"predicted churn probability = {y_proba_test[highest_risk_idx]:.4f}, "
          f"actual churn = {y_test.iloc[highest_risk_idx]}")

    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[highest_risk_idx], max_display=12, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_local_highest_risk.png", dpi=120)
    plt.close(fig)

    print("\nSaved: reports/shap_global_bar.png, reports/shap_global_beeswarm.png, "
          "reports/shap_local_highest_risk.png")
