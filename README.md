# Fraud Detection Dashboard using Machine Learning

## Project Overview
This project builds an end-to-end **Fraud Detection System** using Machine Learning and an interactive **Streamlit Dashboard**.

The objective is to identify fraudulent financial transactions using predictive modeling, explain predictions using SHAP, and provide actionable business insights.

---

## Problem Statement
Fraudulent financial transactions cause major financial losses to organizations.

This project aims to:
- detect fraudulent transactions accurately,
- minimize false positives,
- provide explainable predictions,
- support operational fraud teams using dashboards.

---

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Plotly
- Streamlit
- Joblib

---

## Project Structure

```text
FraudDetection_Meghana M/
│
├── analysis.ipynb
├── train_transaction.csv
├── train_identity.csv
│
├── dashboard/
│   ├── app.py
│   ├── model.pkl
│   ├── explainer.pkl
│   ├── processed_test.csv
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Transaction_Explorer.py
│       └── 3_SHAP_Explainer.py
│
├── charts/
│   ├── shap_summary.png
│   ├── fraud_rate_by_hour.png
│   ├── transaction_amount_distribution.png
│   ├── risk_tier_donut.png
│   ├── precision_recall_curve.png
│   └── interactive_scatter_plot.html
│
├── model_comparison.png
├── summary.pdf
├── requirements.txt
└── README.md
```

---

# Model Performance
Best model used: **XGBoost**

Why:
- strong performance on imbalanced fraud data
- higher precision-recall capability
- captures non-linear fraud patterns well

---

# Streamlit Dashboard Features

## Page 1 — Overview
Displays:
- Total Transactions
- Total Fraud Count
- Detection Rate
- Average Fraud Amount

---

## Page 2 — Transaction Explorer
Features:
- Searchable transaction table
- Filterable transactions
- Live risk score by TransactionID

---

## Page 3 — SHAP Explainer
Features:
- User enters TransactionID
- Plain-English fraud explanation
- Interactive SHAP feature impact chart
- Sidebar filters

---

# Visualizations Included
1. SHAP Global Summary Plot
2. Fraud Rate by Hour of Day
3. Transaction Amount Distribution
4. Risk Tier Donut Chart
5. Precision-Recall Curve
6. Interactive Plotly Scatter Plot

---

# Key Insights
- High transaction amount strongly indicates fraud.
- Late-night transactions show higher fraud risk.
- Abnormal spending ratios are important fraud signals.
- PR-AUC is more reliable than accuracy for fraud detection.

---

# Installation

Clone repository:

```bash
git clone <your-github-repo-link>
cd FraudDetection_Meghana M
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit app:

```bash
streamlit run dashboard/app.py
```

---

# Streamlit Live App


`https://meghana-officialapp-wsv7rpjrwmkndgeu9f8trw.streamlit.app/`



---

# Author
**Meghana M**
