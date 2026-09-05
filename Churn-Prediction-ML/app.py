"""
Stage 8 — Streamlit app: churn risk predictor + plain-English summary.

Ties every prior stage together into something a non-technical customer
support/sales person could actually use:

- Stage 2's `clean_data()` + Stage 7's saved pipeline -> a churn probability
  for one customer, typed in through a form.
- Stage 5's cost-optimal threshold (0.10, saved alongside the model) -> a
  Low/Medium/High risk tier, not just a raw number.
- Stage 6's SHAP approach, applied to this ONE customer -> which of their
  specific attributes are pushing their risk up or down.
- NEW this stage: Gemini (same langchain_google_genai pattern as
  Event-Planner-AI) turns the SHAP breakdown into a short, jargon-free
  explanation for someone who has never heard of SHAP values or logistic
  regression -- "why is this customer at risk, and what should we do."

The chart + SHAP plot need no LLM at all -- they're direct visualizations of
the model's own output. The LLM's only job is translation: numbers and
feature names -> a sentence a retention rep can act on.
"""

import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from preprocessing import load_data, clean_data, split_data
from utils import extract_text, humanize_feature_name

load_dotenv()

MODEL_PATH = "models/churn_pipeline.joblib"

RAW_CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}

# Defaults = the most common value per column in the training data (computed
# once, see learning.md) -- makes the form start on a realistic customer
# instead of an arbitrary first option.
DEFAULTS = {
    "gender": "Male", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "tenure": 29, "MonthlyCharges": 70.35, "SeniorCitizen": False,
}


@st.cache_resource
def load_pipeline_bundle():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["threshold"]


@st.cache_resource
def load_shap_explainer(_pipeline):
    """Background = full training set, transformed the same way the pipeline
    transforms it internally -- same reasoning as Stage 6 (a linear model is
    cheap enough to explain exactly, no need for the default 100-row
    subsample). Leading underscore on `_pipeline` tells st.cache_resource not
    to try hashing the (unhashable) sklearn object itself."""
    raw = load_data()
    cleaned = clean_data(raw)
    X_train, _, _, _ = split_data(cleaned)

    preprocessor = _pipeline.named_steps["preprocessor"]
    classifier = _pipeline.named_steps["classifier"]
    X_train_t = preprocessor.transform(X_train)
    X_train_dense = np.asarray(X_train_t.todense()) if hasattr(X_train_t, "todense") else np.asarray(X_train_t)

    feature_names = list(preprocessor.get_feature_names_out())
    background = shap.maskers.Independent(X_train_dense, max_samples=X_train_dense.shape[0])
    return shap.Explainer(classifier, background, feature_names=feature_names), feature_names


@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        temperature=0.3,
    )


def predict(pipeline, threshold, customer: dict):
    raw_row = pd.DataFrame([customer])
    raw_row["TotalCharges"] = raw_row["MonthlyCharges"] * raw_row["tenure"]
    cleaned_row = clean_data(raw_row)
    proba = float(pipeline.predict_proba(cleaned_row)[0, 1])

    if proba >= 0.5:
        tier = "High"
    elif proba >= threshold:
        tier = "Medium"
    else:
        tier = "Low"
    return proba, tier, cleaned_row


def get_shap_breakdown(pipeline, explainer, feature_names, cleaned_row):
    preprocessor = pipeline.named_steps["preprocessor"]
    row_t = preprocessor.transform(cleaned_row)
    row_dense = np.asarray(row_t.todense()) if hasattr(row_t, "todense") else np.asarray(row_t)
    shap_values = explainer(row_dense)

    contributions = pd.Series(shap_values.values[0], index=feature_names)
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(6)
    return shap_values, top


def build_llm_summary(customer: dict, proba: float, tier: str, top_factors: pd.Series) -> str:
    factor_lines = []
    for name, value in top_factors.items():
        direction = "increases" if value > 0 else "decreases"
        factor_lines.append(f"- {humanize_feature_name(name)} ({direction} risk, impact {abs(value):.2f})")
    factors_text = "\n".join(factor_lines)

    prompt = f"""You are helping a customer support / sales team member understand a churn-risk
prediction for one customer. Do NOT mention SHAP, coefficients, logistic
regression, or any ML jargon. Write for someone with no data science
background.

Customer profile: {customer}
Predicted churn probability: {proba:.0%}
Risk tier: {tier}

Top factors driving this prediction (already ranked by impact):
{factors_text}

Write:
1. A 2-3 sentence plain-English explanation of why this customer is at this
   risk level, referencing the real factors above in everyday language.
2. Two or three concrete, specific retention actions the team could offer
   this particular customer, tied to the actual factors above (not generic
   advice).

Keep it under 150 words total. Address the reader directly as "your team."
"""
    llm = load_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return extract_text(response.content)


st.set_page_config(page_title="Churn risk predictor", layout="wide")
st.title("Customer churn risk predictor")
st.caption(
    "Fill in a customer's profile to predict their churn risk, see what's driving "
    "it, and get a plain-English summary for your team."
)

with st.form("customer_form", border=False):
    with st.container(border=True):
        st.subheader("Customer profile")
        with st.container(horizontal=True):
            gender = st.selectbox("Gender", RAW_CATEGORICAL_OPTIONS["gender"], index=RAW_CATEGORICAL_OPTIONS["gender"].index(DEFAULTS["gender"]))
            senior = st.checkbox("Senior citizen", value=DEFAULTS["SeniorCitizen"])
            partner = st.selectbox("Has a partner", RAW_CATEGORICAL_OPTIONS["Partner"], index=RAW_CATEGORICAL_OPTIONS["Partner"].index(DEFAULTS["Partner"]))
            dependents = st.selectbox("Has dependents", RAW_CATEGORICAL_OPTIONS["Dependents"], index=RAW_CATEGORICAL_OPTIONS["Dependents"].index(DEFAULTS["Dependents"]))

    with st.container(border=True):
        st.subheader("Account")
        with st.container(horizontal=True):
            tenure = st.number_input("Months as a customer (tenure)", min_value=0, max_value=100, value=DEFAULTS["tenure"])
            contract = st.selectbox("Contract", RAW_CATEGORICAL_OPTIONS["Contract"], index=RAW_CATEGORICAL_OPTIONS["Contract"].index(DEFAULTS["Contract"]))
            paperless = st.selectbox("Paperless billing", RAW_CATEGORICAL_OPTIONS["PaperlessBilling"], index=RAW_CATEGORICAL_OPTIONS["PaperlessBilling"].index(DEFAULTS["PaperlessBilling"]))
            payment_method = st.selectbox("Payment method", RAW_CATEGORICAL_OPTIONS["PaymentMethod"], index=RAW_CATEGORICAL_OPTIONS["PaymentMethod"].index(DEFAULTS["PaymentMethod"]))

    with st.container(border=True):
        st.subheader("Services")
        with st.container(horizontal=True):
            phone_service = st.selectbox("Phone service", RAW_CATEGORICAL_OPTIONS["PhoneService"], index=RAW_CATEGORICAL_OPTIONS["PhoneService"].index(DEFAULTS["PhoneService"]))
            multiple_lines = st.selectbox("Multiple lines", RAW_CATEGORICAL_OPTIONS["MultipleLines"], index=RAW_CATEGORICAL_OPTIONS["MultipleLines"].index(DEFAULTS["MultipleLines"]))
            internet_service = st.selectbox("Internet service", RAW_CATEGORICAL_OPTIONS["InternetService"], index=RAW_CATEGORICAL_OPTIONS["InternetService"].index(DEFAULTS["InternetService"]))
        with st.container(horizontal=True):
            online_security = st.selectbox("Online security", RAW_CATEGORICAL_OPTIONS["OnlineSecurity"], index=RAW_CATEGORICAL_OPTIONS["OnlineSecurity"].index(DEFAULTS["OnlineSecurity"]))
            online_backup = st.selectbox("Online backup", RAW_CATEGORICAL_OPTIONS["OnlineBackup"], index=RAW_CATEGORICAL_OPTIONS["OnlineBackup"].index(DEFAULTS["OnlineBackup"]))
            device_protection = st.selectbox("Device protection", RAW_CATEGORICAL_OPTIONS["DeviceProtection"], index=RAW_CATEGORICAL_OPTIONS["DeviceProtection"].index(DEFAULTS["DeviceProtection"]))
        with st.container(horizontal=True):
            tech_support = st.selectbox("Tech support", RAW_CATEGORICAL_OPTIONS["TechSupport"], index=RAW_CATEGORICAL_OPTIONS["TechSupport"].index(DEFAULTS["TechSupport"]))
            streaming_tv = st.selectbox("Streaming TV", RAW_CATEGORICAL_OPTIONS["StreamingTV"], index=RAW_CATEGORICAL_OPTIONS["StreamingTV"].index(DEFAULTS["StreamingTV"]))
            streaming_movies = st.selectbox("Streaming movies", RAW_CATEGORICAL_OPTIONS["StreamingMovies"], index=RAW_CATEGORICAL_OPTIONS["StreamingMovies"].index(DEFAULTS["StreamingMovies"]))

    with st.container(border=True):
        st.subheader("Billing")
        monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=DEFAULTS["MonthlyCharges"], step=0.5)
        st.caption("Total charges are estimated automatically as monthly charges × tenure.")

    submitted = st.form_submit_button("Predict churn risk", icon=":material/insights:")

if submitted:
    customer = {
        "gender": gender, "SeniorCitizen": int(senior), "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
    }

    pipeline, threshold = load_pipeline_bundle()
    proba, tier, cleaned_row = predict(pipeline, threshold, customer)
    explainer, feature_names = load_shap_explainer(pipeline)
    shap_values, top_factors = get_shap_breakdown(pipeline, explainer, feature_names, cleaned_row)

    st.divider()
    tier_color = {"Low": "green", "Medium": "orange", "High": "red"}[tier]

    with st.container(border=True):
        with st.container(horizontal=True):
            st.metric("Predicted churn probability", f"{proba:.0%}")
            with st.container():
                st.caption("Risk tier")
                st.badge(f"{tier} risk", color=tier_color)
            st.metric("Action threshold (Stage 5)", f"{threshold:.0%}")

        chart_df = pd.DataFrame({
            "label": ["This customer"],
            "probability": [proba],
        })
        base = alt.Chart(chart_df).mark_bar(color="#4C72B0").encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Churn probability")),
            y=alt.Y("label:N", title=None),
        )
        threshold_rule = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(
            color="red", strokeDash=[4, 4]
        ).encode(x="threshold:Q")
        st.altair_chart((base + threshold_rule), width="stretch")
        st.caption("Dashed line = the cost-optimal action threshold found in Stage 5.")

    st.subheader("What's driving this prediction")
    # SHAP's waterfall plot has no native Streamlit/Altair equivalent -- kept
    # as the one deliberate matplotlib exception, same as Stage 6.
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Summary for your team")
    with st.spinner("Asking Gemini for a plain-English summary...", show_time=True):
        try:
            summary = build_llm_summary(customer, proba, tier, top_factors)
            with st.container(border=True):
                st.markdown(summary)
        except Exception as e:
            st.error(f"Couldn't reach Gemini for a summary ({e}). The prediction and chart above are still valid.")
