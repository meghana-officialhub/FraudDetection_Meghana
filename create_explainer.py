import shap
import joblib
import pandas as pd

print("Loading model...")
model = joblib.load("dashboard/model.pkl")

print("Loading processed dataset...")
df = pd.read_csv("dashboard/processed_test.csv")

# use current exact columns
feature_names = df.drop(
    columns=["isFraud"],
    errors="ignore"
).columns.tolist()

print("Feature count:", len(feature_names))

print("Creating explainer...")
explainer = shap.TreeExplainer(model)

explainer_data = {
    "explainer": explainer,
    "feature_names": feature_names
}

joblib.dump(
    explainer_data,
    "dashboard/explainer.pkl"
)

print("Done.")