import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# SESSION STATE
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------------------
# PAGE CONFIG + BACKGROUND
# ---------------------------
st.set_page_config(page_title="Churn Prediction", layout="centered")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}

[data-testid="stSidebar"] {
background: rgba(255,255,255,0.1);
backdrop-filter: blur(10px);
}

.block-container {
background: rgba(255,255,255,0.1);
padding: 2rem;
border-radius: 15px;
backdrop-filter: blur(15px);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
# ---------------------------
# USER DATABASE (CSV)
# ---------------------------
def load_users():
    try:
        return pd.read_csv("users.csv")
    except:
        return pd.DataFrame(columns=["username","password"])

def save_user(username, password):
    users = load_users()
    users = pd.concat([users, pd.DataFrame([[username,password]], columns=["username","password"])])
    users.to_csv("users.csv", index=False)

# ---------------------------
# SIDEBAR MENU
# ---------------------------
menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

# ---------------------------
# REGISTER
# ---------------------------
if menu == "Register":
    st.title("📝 Register")

    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")

    if st.button("Register"):
        users = load_users()

        if new_user in users["username"].values:
            st.error("User already exists")
        else:
            save_user(new_user, new_pass)
            st.success("Registration successful! Go to Login")

# ---------------------------
# LOGIN
# ---------------------------
if menu == "Login":
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()

        if ((users["username"] == username) & (users["password"] == password)).any():
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

# ---------------------------
# MAIN APP (ONLY AFTER LOGIN)
# ---------------------------
if st.session_state.logged_in:

    st.title("📊 Customer Churn Prediction App")
    st.markdown("Predict whether a customer will churn")

    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # ---------------------------
    # LOAD & TRAIN MODEL
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
    # PREP INPUT
    # ---------------------------
    input_data = pd.DataFrame([X.mean()])

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly
    input_data["TotalCharges"] = total

    input_data["Contract_One year"] = 1 if contract == "One year" else 0
    input_data["Contract_Two year"] = 1 if contract == "Two year" else 0

    input_data["PaymentMethod_Electronic check"] = 1 if payment == "Electronic check" else 0
    input_data["PaymentMethod_Mailed check"] = 1 if payment == "Mailed check" else 0

    # ---------------------------
    # PREDICTION
    # ---------------------------
    if st.button("🔍 Predict"):
        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("Result")

        st.progress(int(prob * 100))

        if prediction[0] == 1:
            st.error(f"⚠ Customer is likely to CHURN\n\nProbability: {prob:.2f}")
        else:
            st.success(f"✅ Customer is likely to STAY\n\nProbability: {prob:.2f}")