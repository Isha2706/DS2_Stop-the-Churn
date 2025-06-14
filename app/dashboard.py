# app/dashboard.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from io import BytesIO
import shap
import os
import matplotlib.pyplot as plt

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS for themes --------------------
def set_custom_css(theme: str):
    if theme == "Dark":
        st.markdown("""
            <style>
            body, .stApp {
                background-color: #121212 !important;
                color: #ffffff !important;
            }
            .stMarkdown, .stText, .stSubheader, .stTitle, .stHeader, .stRadio label, .stFileUploader label, .stDataFrame {
                color: #ffffff !important;
            }
            .stSidebar, .st-emotion-cache-6qob1r, .st-emotion-cache-1d3w5wq, .css-1d391kg {
                background-color: #1e1e1e !important;
                color: #ffffff !important;
            }
            .stButton>button {
                background-color: #FF4B4B !important;
                color: white !important;
                border: none;
            }
            .stDataFrame table, .stDataFrame td, .stDataFrame th {
                background-color: #1f1f1f !important;
                color: #ffffff !important;
            }
            /* Sidebar widget labels and values */
            .stRadio, .stCheckbox, .stSlider, .stSelectbox, .stNumberInput, .stTextInput {
                color: #ffffff !important;
            }
            .stSlider > div[data-baseweb="slider"] > div {
                color: #ffffff !important;
            }
            .stDownloadButton > button {
                background-color: #FF4B4B !important;
                color: white !important;
                border: none;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body, .stApp {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            .stSidebar {
                background-color: #f0f2f6 !important;
                color: #000000 !important;
            }
            .stButton>button, .stDownloadButton > button {
                background-color: #1f77b4 !important;
                color: white !important;
                border: none;
            }
            </style>
        """, unsafe_allow_html=True)


# -------------------- Sidebar --------------------
st.sidebar.title("Settings")
theme = st.sidebar.radio("Theme Mode", ["Light", "Dark"], horizontal=True)
set_custom_css(theme)

USE_SHAP = st.sidebar.checkbox("Show SHAP Feature Impact", value=True)
thresh = st.sidebar.slider("Churn Threshold", 0.0, 1.0, 0.5)

# -------------------- Load model and scaler --------------------
base = os.path.dirname(__file__)
feature_names = joblib.load(os.path.join(base, "model/feature_names.pkl"))
model = joblib.load(os.path.join(base, "model/model.pkl"))
scaler = joblib.load(os.path.join(base, "model/scaler.pkl"))

# -------------------- UI Header --------------------
st.title("Customer Churn Predictor")
st.markdown("Upload your customer data CSV to infer churn probabilities in real time.")

# -------------------- File upload --------------------
file = st.file_uploader("Upload CSV File", type=["csv"])
if not file:
    st.info("Please upload a test data CSV file to proceed.")
    st.stop()

df = pd.read_csv(file)
st.subheader("Raw Input Data")
st.dataframe(df.head())

# -------------------- Preprocess --------------------
X = pd.get_dummies(df)
X = X.fillna(method='ffill')
for col in feature_names:
    if col not in X.columns:
        X[col] = 0
X = X[feature_names]
X_scaled = scaler.transform(X)

# -------------------- Predict --------------------
probs = model.predict_proba(X_scaled)[:, 1]
df["Churn_Prob"] = probs
df["Churn_Pred"] = (probs >= thresh).astype(int)

# -------------------- Visualizations --------------------
plot_template = "plotly_dark" if theme == "Dark" else "plotly"
bg_color = "#121212" if theme == "Dark" else "#ffffff"
text_color = "#ffffff" if theme == "Dark" else "#000000"

st.subheader("Churn Probability Distribution")
fig_dist = px.histogram(df, x="Churn_Prob", nbins=30, marginal="box", title="Churn Probability Histogram", template=plot_template)
fig_dist.update_layout(
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    font=dict(color=text_color),
    title_font=dict(color=text_color),
    legend=dict(font=dict(color=text_color)),
    xaxis=dict(color=text_color),
    yaxis=dict(color=text_color)
)
st.plotly_chart(fig_dist, use_container_width=True)

st.subheader("Retain vs. Churn")
counts = df["Churn_Pred"].value_counts().rename({0: "Retain", 1: "Churn"})
fig_pie = px.pie(names=counts.index, values=counts.values, title="Churn vs Retain", template=plot_template)
fig_pie.update_layout(
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    font=dict(color=text_color),
    title_font=dict(color=text_color),
    legend=dict(font=dict(color=text_color))
)
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Top 10 At‑Risk Customers")
st.dataframe(df.sort_values("Churn_Prob", ascending=False).head(10))

# -------------------- SHAP --------------------
if USE_SHAP:
    st.subheader("SHAP Feature Impact (Top 100 Samples)")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled[:100])

    # Set background and text color
    bg_color = "#121212" if theme == "Dark" else "#ffffff"
    text_color = "#ffffff" if theme == "Dark" else "#000000"

    # Create SHAP bar plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X.iloc[:100], plot_type="bar", show=False)

    # Update background and axis label colors
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.tick_params(colors=text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)

    # Update spine colors (axes border lines)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    st.pyplot(fig)

# -------------------- Download --------------------
buffer = BytesIO()
df.to_csv(buffer, index=False)
st.download_button("Download Predictions CSV", buffer.getvalue(), "churn_preds.csv", "text/csv")
