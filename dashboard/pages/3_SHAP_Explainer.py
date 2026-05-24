import streamlit as st
import pandas as pd
import joblib
import shap
@st.cache_data
def load_data(path, rows=None):
    return pd.read_csv(path, nrows=rows)

st.title("🔍 SHAP Explanation Engine")

# Load model + explainer
model = joblib.load("dashboard/model.pkl")
explainer = shap.TreeExplainer(model)
feature_names = model.feature_names_in_

# Load processed test data
df = pd.read_csv("dashboard/processed_test.csv", nrows=5000)

# ---------------------------
# 🌟 SIDEBAR FILTERS
# ---------------------------
st.sidebar.header("🔎 Filter Transactions")

# Product type filter
product_filter = st.sidebar.multiselect(
    "Product Code (ProductCD)", 
    options=sorted(df["ProductCD"].dropna().unique().tolist()),
    default=None
)

# Card network filter
card_filter = st.sidebar.multiselect(
    "Card Type (card4)", 
    options=sorted(df["card4"].dropna().unique().tolist()),
    default=None
)

# Email domain filter
email_filter = st.sidebar.multiselect(
    "Email Domain (P_emaildomain)", 
    options=sorted(df["P_emaildomain"].dropna().unique().tolist()),
    default=None
)

# Amount slider filter
min_amt, max_amt = int(df["TransactionAmt"].min()), int(df["TransactionAmt"].max())

amount_range = st.sidebar.slider(
    "Transaction Amount Range", 
    min_amt, max_amt, 
    (min_amt, max_amt)
)

# Apply filters
filtered_df = df.copy()

if product_filter:
    filtered_df = filtered_df[filtered_df["ProductCD"].isin(product_filter)]

if card_filter:
    filtered_df = filtered_df[filtered_df["card4"].isin(card_filter)]

if email_filter:
    filtered_df = filtered_df[filtered_df["P_emaildomain"].isin(email_filter)]

filtered_df = filtered_df[
    (filtered_df["TransactionAmt"] >= amount_range[0]) &
    (filtered_df["TransactionAmt"] <= amount_range[1])
]

st.sidebar.success(f"Filtered rows: {len(filtered_df)}")

# ---------------------------
# MAIN UI
# ---------------------------

st.write("Enter a TransactionID from the filtered dataset.")

# Show valid IDs
with st.expander("📌 View Available TransactionIDs"):
    st.write(filtered_df["TransactionID"].tolist())

transaction_id = st.text_input("TransactionID")

if st.button("Explain Prediction"):
    if transaction_id.strip() == "":
        st.error("Please enter a TransactionID.")
    else:
        transaction_id = int(float(transaction_id))

        if transaction_id not in filtered_df["TransactionID"].values:
            st.error("❌ TransactionID not found in the filtered dataset.")
        else:
            st.success("Transaction found!")

            # Extract row
            row = df[df["TransactionID"] == transaction_id].iloc[0]

            # SHAP requires all model features
            X = row.drop(labels=["isFraud"], errors="ignore").to_frame().T

            shap_values = explainer(X)

            shap_dict = dict(
                zip(feature_names, shap_values.values[0, 1:, ])  # skip TransactionID
            )

            # Sort by magnitude
            top_features = sorted(
                shap_dict.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:5]

            # ------------------------------
            # 🧠 Plain-English Explanation
            # ------------------------------
            st.subheader("🧠 Plain-English Explanation")
            for feature, value in top_features:
                direction = "increased" if value > 0 else "decreased"
                st.write(f"- **{feature}** {direction} the fraud probability (impact: `{value:.4f}`)")

            st.info("Model prediction explained using SHAP values.")

            # ------------------------------
            # 📊 Plotly Bar Chart (Top Features)
            # ------------------------------
            import plotly.graph_objects as go

            st.subheader("📊 Top Feature Impacts (Interactive)")

            feature_names_plot = [item[0] for item in top_features]
            feature_values_plot = [item[1] for item in top_features]

            fig = go.Figure(
                go.Bar(
                    x=feature_values_plot,
                    y=feature_names_plot,
                    orientation='h',
                    hovertemplate='<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>',
                )
            )

            fig.update_layout(
                title="Top SHAP Feature Contributions",
                xaxis_title="SHAP Impact (Positive → Higher Fraud Probability)",
                yaxis_title="Features",
                height=400,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)