
import streamlit as st
import pandas as pd
import pickle

st.title("Loan Approval Prediction")
st.write("Enter applicant details")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income = st.number_input("Annual Income", min_value=0, value=500000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=1000000)
loan_term = st.number_input("Loan Term (Months)", min_value=1, value=12)
cibil = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
residential = st.number_input("Residential Assets Value", min_value=0, value=100000)
commercial = st.number_input("Commercial Assets Value", min_value=0, value=100000)
luxury = st.number_input("Luxury Assets Value", min_value=0, value=100000)
bank = st.number_input("Bank Asset Value", min_value=0, value=100000)

if st.button("Predict Loan Status"):
    row = {
        "no_of_dependents": dependents,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil,
        "residential_assets_value": residential,
        "commercial_assets_value": commercial,
        "luxury_assets_value": luxury,
        "bank_asset_value": bank,
    }

    if "education_ Not Graduate" in feature_columns:
        row["education_ Not Graduate"] = int(education == "Not Graduate")
    if "education_Not Graduate" in feature_columns:
        row["education_Not Graduate"] = int(education == "Not Graduate")
    if "self_employed_ Yes" in feature_columns:
        row["self_employed_ Yes"] = int(self_employed == "Yes")
    if "self_employed_Yes" in feature_columns:
        row["self_employed_Yes"] = int(self_employed == "Yes")

    input_df = pd.DataFrame([row])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
        
