import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Continuous Behavioral Authentication",
    page_icon="🛡️",
    layout="wide"
)

# ------------------ Load Model ------------------
model = joblib.load("balanced_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ Load Dataset ------------------
df = pd.read_csv("labelled data.csv")

# Features and labels
X = df.drop(['subject', 'sessionIndex', 'rep', 'Label'], axis=1)
y = df['Label']

# Scale features
X_scaled = scaler.transform(X)

# ------------------ Sidebar ------------------
st.sidebar.title("Select User Sample")

sample = st.sidebar.slider(
    "Choose Dataset Row",
    0,
    len(df)-1,
    0
)

# Selected sample
sample_data = X_scaled[sample].reshape(1,-1)

# Prediction
prediction = model.predict(sample_data)[0]

# Confidence
confidence = np.max(model.predict_proba(sample_data))*100

# Actual label
actual = y.iloc[sample]

# Risk Score
risk = round(100-confidence,2)

# Authentication Status
if prediction == 1:
    status = "✅ Authenticated"
    risk_text = "Low Risk"
else:
    status = "❌ Imposter"
    risk_text = "High Risk"

# ------------------ Title ------------------
st.title("🛡️ Continuous Behavioral Authentication System")
st.markdown("### AI Powered Continuous User Monitoring Dashboard")

st.divider()

# ------------------ Top Metrics ------------------

col1,col2,col3,col4 = st.columns(4)

col1.metric("Accuracy","97.79%")
col2.metric("FAR","2.10%")
col3.metric("FRR","7.50%")
col4.metric("Model Status","Active ✅")

st.divider()

# ------------------ Live Session ------------------

left,right = st.columns([2,1])

with left:

    st.subheader("👤 Live User Session")

    st.info(f"""
Current User : User_{df.iloc[sample]['subject']}

Authentication Status : {status}

Model Confidence : {confidence:.2f}%

Actual Label : {actual}
""")

with right:

    st.subheader("⚠️ Risk Score")

    st.metric("Risk Level",f"{risk:.2f}%")

    if risk<30:
        st.success(risk_text)
    elif risk<60:
        st.warning("Medium Risk")
    else:
        st.error(risk_text)

st.divider()

# ------------------ Prediction ------------------

st.subheader("🤖 Model Prediction")

if prediction==actual:
    st.success("Prediction matches the actual label.")
else:
    st.error("Prediction does not match the actual label.")

st.write("Predicted Label :",prediction)
st.write("Actual Label :",actual)

st.divider()

# ------------------ Sample Features ------------------

st.subheader("Selected User Features")

st.dataframe(
    pd.DataFrame(
        X.iloc[sample].T,
        columns=["Value"]
    )
)

st.divider()

# ------------------ Authentication Log ------------------

st.subheader("Authentication Log")

log=[]

for i in range(max(0,sample-5),sample+1):

    pred=model.predict(X_scaled[i].reshape(1,-1))[0]

    if pred==1:
        s="Authenticated"
    else:
        s="Imposter"

    log.append({
        "Sample":i,
        "Subject":df.iloc[i]["subject"],
        "Prediction":s
    })

st.table(pd.DataFrame(log))
