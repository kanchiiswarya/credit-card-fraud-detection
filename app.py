import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("💳 Fraud Detection Demo")
st.write("Upload transaction data or enter values manually to check fraud probability.")

model = joblib.load('fraud_model.pkl')

uploaded_file = st.file_uploader("Upload a CSV of transactions", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", data.head())

    if st.button("Predict Fraud"):
        preds = model.predict(data)
        probs = model.predict_proba(data)[:, 1]
        data['Fraud_Prediction'] = preds
        data['Fraud_Probability'] = probs
        st.write(data[['Fraud_Prediction', 'Fraud_Probability']])

        # SHAP explanation for first row
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data.drop(columns=['Fraud_Prediction','Fraud_Probability'], errors='ignore'))
        st.write("SHAP explanation for first transaction:")
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value, shap_values[0], 
                         data.drop(columns=['Fraud_Prediction','Fraud_Probability'], errors='ignore').iloc[0], 
                         matplotlib=True, show=False)
        st.pyplot(fig)
