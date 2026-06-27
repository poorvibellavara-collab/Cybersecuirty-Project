import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="Continuous Behavioral Authentication",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- Load Files ----------------

model = joblib.load("balanced_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("labelled data.csv")

# ---------------- Prepare Data ----------------

X = df.drop(["subject", "sessionIndex", "rep", "Label"], axis=1)
y = df["Label"]

X_scaled = scaler.transform(X)

# ---------------- Overall Model Performance ----------------

predictions = model.predict(X_scaled)

accuracy = accuracy_score(y, predictions)

tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

far = (fp / (fp + tn)) * 100
frr = (fn / (fn + tp)) * 100

# ---------------- Sidebar ----------------

st.sidebar.title("User Selection")

sample = st.sidebar.slider(
    "Select Sample",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

sample_data = X_scaled[sample].reshape(1, -1)

prediction = model.predict(sample_data)[0]

probability = model.predict_proba(sample_data)[0]
confidence = np.max(probability) * 100

actual = y.iloc[sample]
subject = df.iloc[sample]["subject"]

# ---------------- Authentication Status ----------------

if prediction == 1:
    status = "✅ Authenticated"
else:
    status = "❌ Imposter"

# ---------------- Risk Logic ----------------

if prediction == 1:
    risk = 100 - confidence

    if risk < 30:
        risk_status = "Low Risk"
    elif risk < 60:
        risk_status = "Medium Risk"
    else:
        risk_status = "High Risk"

else:
    risk = confidence
    risk_status = "🚨 High Risk - Imposter Detected"

# ---------------- Dashboard ----------------

st.title("🛡️ Continuous Behavioral Authentication System")
st.markdown("### AI Powered Continuous User Monitoring Dashboard")

st.divider()

# ---------------- Top Metrics ----------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy*100:.2f}%")
c2.metric("FAR", f"{far:.2f}%")
c3.metric("FRR", f"{frr:.2f}%")
c4.metric("Model Status", "Active ✅")

st.divider()

# ---------------- Live Session ----------------

left, right = st.columns([2, 1])

with left:

    st.subheader("👤 Live User Session")

    st.info(f"""
Current User : {subject}

Authentication Status : {status}

Prediction : {prediction}

Actual Label : {actual}

Confidence Score : {confidence:.2f}%
""")

with right:

    st.subheader("⚠️ Risk Analysis")

    st.metric("Risk Level", f"{risk:.2f}%")

    if prediction == 1:

        if risk < 30:
            st.success(risk_status)

        elif risk < 60:
            st.warning(risk_status)

        else:
            st.error(risk_status)

    else:
        st.error(risk_status)

st.divider()

# ---------------- Prediction Summary ----------------

st.subheader("Prediction Summary")

if prediction == actual:
    st.success("Prediction matches the actual label.")
else:
    st.warning("Prediction does not match the actual label.")

# ---------------- Selected Features ----------------

st.subheader("Selected User Features")

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Value": X.iloc[sample].values
})

st.dataframe(feature_df, use_container_width=True)

st.divider()

# ---------------- Authentication Log ----------------

st.subheader("Authentication Log")

log = []

start = max(0, sample - 5)
end = min(len(df), sample + 1)

for i in range(start, end):

    row = X_scaled[i].reshape(1, -1)

    pred = model.predict(row)[0]

    conf = np.max(model.predict_proba(row)) * 100

    if pred == 1:
        auth = "Authenticated"
    else:
        auth = "Imposter"

    log.append({
        "Sample": i,
        "User": df.iloc[i]["subject"],
        "Prediction": auth,
        "Confidence": f"{conf:.2f}%"
    })

st.dataframe(pd.DataFrame(log), use_container_width=True)
