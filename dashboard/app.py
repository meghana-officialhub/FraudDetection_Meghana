import streamlit as st

st.set_page_config(
    page_title="Fraud Operations Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Operations Dashboard")
st.markdown("### Credit Card Fraud Detection & Risk Monitoring")

st.write("""
Welcome to the Fraud Operations Dashboard.

Use the sidebar to navigate between pages:

1. Overview
2. Transaction Explorer
3. SHAP Explainer
""")