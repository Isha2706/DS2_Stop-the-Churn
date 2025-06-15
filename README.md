# Fintech Customer Churn Predictor

## Overview

Customer churn is a critical issue in the fintech industry—every lost customer directly affects revenue. This project delivers a **churn prediction tool and interactive dashboard** that enables stakeholders to:

- Upload customer data
- Analyze churn risk
- Identify at-risk users
- Take data-backed actions to retain them

It combines predictive modeling with actionable visual insights, wrapped in a modern, user-friendly dashboard.

## Project Structure

```bash
.
├── app/
│   ├── dashboard.py             # Streamlit dashboard app
│   └── model/                   # Model files
│       ├── feature_names.pkl
│       ├── model.pkl
│       ├── scaler.pkl
│       ├── X_test_scaled.pkl
│       ├── X_train_scaled.pkl
│       ├── Y_test.pkl
│       └── Y_train.pkl
├── data/
│   ├── telco_train.csv          # Training data
│   └── telco_test.csv           # Hidden test data
├── notebooks
│   ├── 01_data_cleaning.ipynb         # Cleaning and Preparing the data
│   ├── 02_model_training.ipynb        # Training the churn prediction model
│   └── 03_model_evaluation.ipynb      # Analyzing feature importance and using SHAP
├── output
│   └── churn_preds.csv           # Result of test data
├── requirements.txt
└── README.md
```

## Setup Instructions

> Ensure Python 3.8+ is installed.

### 1. Clone the repository
```bash
git clone https://github.com/Isha2706/DS2_Stop-the-Churn.git
cd DS2_Stop-the-Churn
```

### 2. Create virtual environment and activate
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit app
```bash
streamlit run app/dashboard.py
```

## Prediction Approach

- **Model Used:** XGBoost Classifier
- **Target:** Predict whether a customer will churn in the next 30 days
- **Evaluation Metric:** AUC-ROC
- **Preprocessing:**
  - Missing value imputation (forward fill)
  - One-hot encoding of categorical features
  - Feature scaling using `StandardScaler`
 
## Model Performance

- **Model:** XGBoost (XGBClassifier)
- **Optimization Metric:** AUC-ROC
- **Explainability:** SHAP summary plots and individual feature impact for transparency
- **Threshold Customization:** Adjustable churn cutoff via the dashboard

| Metric   | Value                                   |
| -------- | --------------------------------------- |
| AUC-ROC  | 0.89+                                   |
| Accuracy | \~85%                                   |
| Recall   | High emphasis to reduce false negatives |

## Dashboard Features

| Feature                     | Description                             |
| --------------------------- | --------------------------------------- |
| CSV Upload                  | Upload behavioral/transactional data    |
| Churn Probability Histogram | Visualizes predicted risk               |
| Retain vs Churn Pie Chart   | Easy understanding of current risk      |
| Top 10 At-Risk Customers    | Ranked by churn probability             |
| SHAP Feature Impact         | Understand why the model predicts churn |
| Download Predictions        | Get all results in CSV                  |
| Dark Mode                   | Switch between Light and Dark Themes    |

## External Files

- Google Drive (View-only):
  - [Raw Data](https://drive.google.com/drive/folders/14tvik6DvLQu6DvZyzO9YvvUJMDB9jdtp?usp=sharing)
  - [Video Presentation](link)

## Extras

- **SHAP Explainability:** Helps understand feature influence per prediction
- **Theme Toggle:** Supports dark/light mode for better UX
- **Inline Explanations:** SHAP insights embedded with top 10 customer table

## Add-ons

- [Live dashboard](https://ds2stop-the-churn-atsqmpwtfahrwjsufa7njo.streamlit.app/)
- [Jupyter Notebook with EDA + Model Training](https://github.com/Isha2706/DS2_Stop-the-Churn/tree/main/notebooks)

## Author

Isha Maheshwari
- [Email Me](mailto:ishamaheshwari2003@gmail.com)
- [GitHub](https://github.com/Isha2706/)
