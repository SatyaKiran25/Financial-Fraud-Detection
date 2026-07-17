import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title= "Financial Fraud Detection",
    page_icon= "💳",
    layout = "wide")


# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

print(BASE_DIR)
print(MODEL_PATH)

# Load model
model = joblib.load(MODEL_PATH)

st.title("💳 Financial Fraud Detection Dashboard")
st.markdown("---")

st.write(
    """
    This application predicts whether a credit card transaction
    is **Fraudulent** or **Legitimate** using a trained Machine Learning model.
    """
)

st.sidebar.title("Amount")
st.sidebar.info(
    """
    Project 1

    Financial Fraud Detection

    Model:
    Random Forest

    Framework:
    Streamlit
    """
)

left, right = st.columns(2)
with left :
    st.subheader("Transaction Details")

    amount = st.number_input(
        "Transaction Amount",
        min_value = 0.0,
        value = 100.0
    )

# Sample transaction
sample = pd.DataFrame([{
    "Time": 0,
    "V1": 0,
    "V2": 0,
    "V3": 0,
    "V4": 0,
    "V5": 0,
    "V6": 0,
    "V7": 0,
    "V8": 0,
    "V9": 0,
    "V10": 0,
    "V11": 0,
    "V12": 0,
    "V13": 0,
    "V14": 0,
    "V15": 0,
    "V16": 0,
    "V17": 0,
    "V18": 0,
    "V19": 0,
    "V20": 0,
    "V21": 0,
    "V22": 0,
    "V23": 0,
    "V24": 0,
    "V25": 0,
    "V26": 0,
    "V27": 0,
    "V28": 0,
    "Amount": amount
}])

if st.button("Predict"):

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)

    if prediction[0] == 1:
        st.error("🚨 Fraud Transaction")
    else:
        st.success("✅ Legitimate Transaction")

    st.write("Prediction Probability")
    fraud_probability = probability[0][1] * 100

    st.metric(
        "Fraud Probability",
        f"{fraud_probability:.2f}%"
    )

    st.markdown("---")

    st.caption(
    "Model: Random Forest Classifier | Developed using Scikit-Learn and Streamlit"
    )
