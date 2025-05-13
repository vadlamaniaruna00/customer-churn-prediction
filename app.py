# Streamlit Front-End for Enhanced Customer Churn Prediction (100 Marks Version)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model only
model = joblib.load("xgb_churnn_model.pkl")


# Define the expected feature names manually (based on training data)
feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
                 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
                 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes',
                 'Contract_One year', 'Contract_Two year', 'PaymentMethod_Credit card (automatic)',
                 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f9f9f9; font-family: 'Segoe UI', sans-serif;}
    h1, h2, h3, h4 {color: #2c3e50;}
    </style>
""", unsafe_allow_html=True)

st.title("📞 Customer Churn Prediction Portal")
st.markdown("This app helps predict whether a telecom customer will churn based on their profile, services, and billing details.")

# Sidebar info
st.sidebar.header("📌 About")
st.sidebar.info("Built by Aruna • MBA – Business Analytics")
st.sidebar.success("Accuracy: ~84.4%\n\nModel: XGBoost + Logistic Regression")

# Input sections
st.header("🧾 Enter Customer Details")
col1, col2 = st.columns(2)

# Customer Information
gender = col1.selectbox("Gender", ["Male", "Female"])
senior = col2.selectbox("Senior Citizen", [0, 1])
partner = col1.selectbox("Has Partner", ["Yes", "No"])
dependent = col2.selectbox("Has Dependents", ["Yes", "No"])

# Services
st.subheader("📡 Services Subscribed")
col3, col4 = st.columns(2)

phone_service = col3.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = col4.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = col3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
streaming_tv = col4.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = col3.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
tech_support = col4.selectbox("Tech Support", ["No", "Yes", "No internet service"])
online_security = col3.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = col4.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = col3.selectbox("Device Protection", ["No", "Yes", "No internet service"])


# Billing & Contract
st.subheader("💳 Billing & Contract")
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 9000.0, 3000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Build input dataframe
input_dict = {
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': senior,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependent == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone_service == 'Yes' else 0,
    'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
    'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
    'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
    'InternetService_No': 1 if internet_service == 'No' else 0,
    'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
    'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
    'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
    'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
    'Contract_One year': 1 if contract == 'One year' else 0,
    'Contract_Two year': 1 if contract == 'Two year' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
    'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
    'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
    'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
    'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
    'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
    'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
    'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0
}

if st.button("🔍 Predict Churn"):
    with st.spinner("Running Prediction..."):

        # Force fresh DataFrame creation inside the button
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=[
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
            'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
            'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_Fiber optic', 'InternetService_No',
            'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
            'OnlineBackup_No internet service', 'OnlineBackup_Yes',
            'DeviceProtection_No internet service', 'DeviceProtection_Yes',
            'TechSupport_No internet service', 'TechSupport_Yes',
            'StreamingTV_No internet service', 'StreamingTV_Yes',
            'StreamingMovies_No internet service', 'StreamingMovies_Yes',
            'Contract_One year', 'Contract_Two year',
            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check'
        ], fill_value=0)

        st.write("🔍 Input DF Preview", input_df)
        
        prob = model.predict_proba(input_df)[0][1]
        threshold = 0.37
        prediction = 1 if prob > threshold else 0

        
    if prediction == 1:
        st.error(f"""
        📢 **Prediction: Churn**

        🧠 **Churn Probability:** {prob:.2%}
        """)
    else:
        st.success(f"""
        📢 **Prediction: Stay**

        🧠 **Churn Probability:** {prob:.2%}
        """)


