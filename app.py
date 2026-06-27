import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

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

X = df.drop(["subject","sessionIndex","rep","Label"], axis=1)
y = df["Label"]

X_scaled = scaler.transform(X)

# ---------------- Predictions ----------------

predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)

# ---------------- Performance Metrics ----------------

accuracy = accuracy_score(y, predictions)

tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

far = (fp / (fp + tn)) * 100
frr = (fn / (fn + tp)) * 100

# ---------------- Sidebar ----------------

st.sidebar.header("User Selection")

sample = st.sidebar.slider(
    "Choose Sample",
    0,
    len(df)-1,
    0
)

sample_data = X_scaled[sample].reshape(1,-1)

prediction = model.predict(sample_data)[0]

confidence = np.max(model.predict_proba(sample_data))*100

actual = y.iloc[sample]

risk = 100-confidence

subject = df.iloc[sample]["subject"]

# ---------------- Status ----------------

if prediction == 1:
    status = "✅ Authenticated"
else:
    status = "❌ Imposter"

if risk < 30:
    risk_status = "Low Risk"
elif risk < 60:
    risk_status = "Medium Risk"
else:
    risk_status = "High Risk"

# ---------------- Dashboard ----------------

st.title("🛡️ Continuous Behavioral Authentication System")

st.markdown("### AI Powered Continuous User Monitoring Dashboard")

st.divider()

# ---------------- Top Metrics ----------------

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{accuracy*100:.2f}%"
)

c2.metric(
    "FAR",
    f"{far:.2f}%"
)

c3.metric(
    "FRR",
    f"{frr:.2f}%"
)

c4.metric(
    "Model Status",
    "Active ✅"
)

st.divider()

# ---------------- User Session ----------------

left,right = st.columns([2,1])

with left:

    st.subheader("Live User Session")

    st.info(f"""
Current User : User_{subject}

Authentication Status : {status}

Prediction : {prediction}

Actual Label : {actual}

Confidence Score : {confidence:.2f}%
""")

with right:

    st.subheader("Risk Analysis")

    st.metric(
        "Risk Level",
        f"{risk:.2f}%"
    )

    if risk < 30:
        st.success(risk_status)
    elif risk < 60:
        st.warning(risk_status)
    else:
        st.error(risk_status)

st.divider()

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

rows=[]

start=max(0,sample-5)

end=min(len(df),sample+1)

for i in range(start,end):

    pred=model.predict(X_scaled[i].reshape(1,-1))[0]

    conf=np.max(
        model.predict_proba(
            X_scaled[i].reshape(1,-1)
        )
    )*100

    rows.append({
        "Sample":i,
        "User":df.iloc[i]["subject"],
        "Prediction":"Authenticated" if pred==1 else "Imposter",
        "Confidence":f"{conf:.2f}%"
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)
