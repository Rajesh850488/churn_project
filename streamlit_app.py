import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("churn_model.pkl", "rb"))

st.title("Customer Churn Prediction App")

tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict"):
    features = np.array([[tenure, monthly_charges, total_charges]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("Customer is likely to CHURN ❌")
    else:
        st.success("Customer will STAY ✔️")
