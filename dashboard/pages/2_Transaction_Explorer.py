import streamlit as st
import pandas as pd
import joblib

st.title("🔍 Transaction Explorer")

# Load model
model = joblib.load("dashboard/model.pkl")

# Load processed data (NO isFraud inside this file)
processed = pd.read_csv("dashboard/processed_test.csv").head(1000)

# Load raw data (used ONLY for display + Fraud filter)
raw = pd.read_csv("train_transaction.csv").head(1000)

# Keep mapping
raw["row_index"] = raw.index  # align with processed

# Sidebar filter
fraud_filter = st.sidebar.selectbox(
    "Filter by Fraud",
    ["All", "Fraud Only", "Non-Fraud Only"]
)

filtered_raw = raw.copy()

if fraud_filter == "Fraud Only":
    filtered_raw = filtered_raw[filtered_raw["isFraud"] == 1]
elif fraud_filter == "Non-Fraud Only":
    filtered_raw = filtered_raw[filtered_raw["isFraud"] == 0]

st.subheader("Searchable & Filterable Transactions")
st.dataframe(filtered_raw.drop(columns=["row_index"]))

# Live Prediction
st.subheader("Live Risk Score by TransactionID")

tx_id = st.number_input(
    "Enter TransactionID",
    min_value=int(raw["TransactionID"].min()),
    max_value=int(raw["TransactionID"].max()),
    step=1
)

if st.button("Predict Risk"):
    match = raw[raw["TransactionID"] == tx_id]

    if len(match) == 0:
        st.error("TransactionID not found")
    else:
        row_idx = match["row_index"].values[0]

        # Use processed row WITHOUT trying to drop isFraud
        X = processed.iloc[[row_idx]]

        prob = model.predict_proba(X)[0][1]

        st.success(f"Fraud Risk Score: {prob:.2%}")