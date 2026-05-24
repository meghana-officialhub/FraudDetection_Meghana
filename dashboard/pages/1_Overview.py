import streamlit as st
import pandas as pd
@st.cache_data
def load_data(path, rows=None):
    return pd.read_csv(path, nrows=rows)

st.title("📊 Overview")

# Load RAW data
df = pd.read_csv("train_transaction.csv", nrows=50000)

# Metrics
total_transactions = len(df)
total_fraud = df["isFraud"].sum()
detection_rate = (total_fraud / total_transactions) * 100
avg_fraud_amount = df[df["isFraud"] == 1]["TransactionAmt"].mean()

# Display metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Total Fraud", f"{total_fraud:,}")
col3.metric("Detection Rate", f"{detection_rate:.2f}%")
col4.metric("Avg Fraud Amount", f"${avg_fraud_amount:.2f}")