# app/dashboard.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from io import BytesIO
import shap
import os
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(
    page_title="Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
USE_SHAP = st.sidebar.checkbox("Show SHAP Feature Impact", value=True)
thresh = st.sidebar.slider("Churn Threshold", 0.0, 1.0, 0.5)

# Load model and scaler
base = os.path.dirname(__file__)
feature_names = joblib.load(os.path.join(base, "model/feature_names.pkl"))
model = joblib.load(os.path.join(base, "model/model.pkl"))
scaler = joblib.load(os.path.join(base, "model/scaler.pkl"))

# UI Header
st.title("💡 Fintech Customer Churn Predictor")
st.markdown("Upload your customer data CSV to infer churn probabilities in real time.")

# File upload
file = st.file_uploader("📤 Upload CSV File", type=["csv"])
if not file:
    st.info("Please upload a test data CSV file to proceed.")
    st.stop()

# Load input data
df = pd.read_csv(file)
st.markdown("### 🧾 Raw Input Data")
st.dataframe(df.head())

# Preprocess: dummy encode + align with training columns
X = pd.get_dummies(df)
X = X.fillna(method='ffill')

# Add missing columns and ensure correct order
for col in feature_names:
    if col not in X.columns:
        X[col] = 0
X = X[feature_names]  # Keep only required columns in order

# Scale
X_scaled = scaler.transform(X)

# Predict
probs = model.predict_proba(X_scaled)[:, 1]
df["Churn_Prob"] = probs
df["Churn_Pred"] = (probs >= thresh).astype(int)

# Visuals
st.subheader("📊 Churn Probability Distribution")
fig_dist = px.histogram(df, x="Churn_Prob", nbins=30, marginal="box", title="Churn Probability Histogram")
st.plotly_chart(fig_dist, use_container_width=True)

st.subheader("🥧 Retain vs. Churn")
counts = df["Churn_Pred"].value_counts().rename({0: "Retain", 1: "Churn"})
fig_pie = px.pie(names=counts.index, values=counts.values, title="Churn vs Retain")
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("🚩 Top 10 At‑Risk Customers")
st.dataframe(df.sort_values("Churn_Prob", ascending=False).head(10))

# SHAP
if USE_SHAP:
    st.subheader("🔍 SHAP Feature Impact (Top 100 Samples)")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled[:100])
    
    # Capture SHAP plot in a matplotlib figure
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X.iloc[:100], plot_type="bar", show=False)
    st.pyplot(fig)

# Download button
buffer = BytesIO()
df.to_csv(buffer, index=False)
st.download_button("📥 Download Predictions CSV", buffer.getvalue(), "churn_preds.csv", "text/csv")
