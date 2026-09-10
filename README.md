# 💳 Credit Card Fraud Detection

A machine learning system to detect fraudulent credit card transactions, addressing severe class imbalance using SMOTE, with model explainability via SHAP and a live interactive demo built with Streamlit.

## 🚀 Live Demo

**Try it here:** [credit-card-fraud-detection-vywwwheq22vip2w9wp2mhs.streamlit.app](https://credit-card-fraud-detection-vywwwheq22vip2w9wp2mhs.streamlit.app/)

Upload a CSV of transactions and the app returns:
- Fraud prediction (0/1) and probability score per transaction
- SHAP-based explanation of why a transaction was flagged
  
## 🎯 Problem

Credit card fraud detection is a classic imbalanced classification problem. In this dataset of **284,807 transactions**, only **492 (0.17%)** are fraudulent — meaning a naive model predicting "not fraud" every time would score 99.8% accuracy while catching zero fraud. This project focuses on building a model that actually catches fraud while minimizing false alarms, and explaining *why* it flags what it flags.

## 📊 Dataset

- Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions, 31 features (V1–V28 are PCA-transformed for privacy, plus Time, Amount, Class)
- Highly imbalanced: 0.17% fraud rate

## 🔍 Key Insight from EDA

Fraudulent transactions had a **lower median amount (₹9.25)** than normal transactions (₹22.00), despite a higher mean — suggesting fraudsters test stolen cards with small transactions before attempting larger ones, a known real-world "card testing" pattern.

## 🛠️ Approach

1. **Train/test split** with stratification to preserve the real-world fraud ratio in the test set
2. **SMOTE** (Synthetic Minority Oversampling) applied **only on training data** — to avoid data leakage into the test set
3. **Baseline model**: Logistic Regression
4. **Main model**: XGBoost — chosen for its strength on tabular, imbalanced data
5. **Explainability**: SHAP values to interpret feature-level contributions to each prediction
6. **Deployment**: Streamlit app for interactive upload → predict → explain

## 📈 Results

| Metric (Fraud class) | Logistic Regression | XGBoost |
|---|---|---|
| Precision | 0.12 | **0.77** |
| Recall | 0.90 | **0.86** |
| F1-score | 0.21 | **0.81** |
| ROC-AUC | 0.976 | **0.983** |

XGBoost dramatically improved precision (12% → 77%) while maintaining strong recall (86%) — meaning far fewer false alarms for nearly the same fraud-catching ability. This trade-off matters in practice: investigating false positives has a real operational cost for fraud teams.

## 🧠 Explainability (SHAP)

SHAP analysis identified **V14** as the most influential feature, with low V14 values strongly associated with fraud predictions. Since features are PCA-anonymized for privacy, exact business meaning isn't available — but the model's decision logic can still be quantified and explained per-transaction, which is critical for real-world deployment where flagged transactions often require justification.

## 🚀 Live Demo

The Streamlit app allows uploading a CSV of transactions and returns:
- Fraud prediction (0/1) and probability score per transaction
- SHAP-based explanation of why a transaction was flagged

## 🧰 Tech Stack

Python · pandas · scikit-learn · imbalanced-learn · XGBoost · SHAP · Streamlit · Google Colab

## 📁 Files

- `fraud_detection.ipynb` — full analysis, from EDA to model training and SHAP
- `app.py` — Streamlit application code

## 🔮 Future Improvements

- Hyperparameter tuning (grid/random search on XGBoost)
- Compare SMOTE against class-weighting as an alternative imbalance strategy
- Deploy permanently (e.g. Streamlit Community Cloud) instead of ngrok tunnel
