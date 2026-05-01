import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("Predict whether a customer will churn based on their details")

# ---------------------------
# Load & Train Model
# ---------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

model = RandomForestClassifier()
model.fit(X, y)

# ---------------------------
# USER INPUTS
# ---------------------------
st.header("🧾 Enter Customer Details")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# ---------------------------
# Prepare Input
# ---------------------------
input_data = pd.DataFrame([X.mean()])

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly
input_data["TotalCharges"] = total

# Contract encoding
input_data["Contract_One year"] = 1 if contract == "One year" else 0
input_data["Contract_Two year"] = 1 if contract == "Two year" else 0

# Payment encoding
input_data["PaymentMethod_Electronic check"] = 1 if payment == "Electronic check" else 0
input_data["PaymentMethod_Mailed check"] = 1 if payment == "Mailed check" else 0

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if prediction[0] == 1:
        st.error(f"⚠ Customer is likely to CHURN\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Customer is likely to STAY\n\nProbability: {prob:.2f}")