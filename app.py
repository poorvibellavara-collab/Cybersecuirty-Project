import streamlit as st

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Continuous Behavioral Authentication",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Continuous Behavioral Authentication System")
st.markdown("### AI Powered Continuous User Monitoring Dashboard")
st.divider()

# ------------------ Top Metrics ------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "97.79%")
col2.metric("FAR", "2.10%")
col3.metric("FRR", "7.50%")
col4.metric("Model Status", "Active ✅")

st.divider()

# ------------------ Live Session ------------------
left, right = st.columns([2,1])

with left:
    st.subheader("👤 Live User Session")

    st.info("""
Current User : User_01

Authentication Status : ✅ Authenticated

Confidence Score : 97%

Session Duration : 00:18:42
""")

with right:
    st.subheader("⚠️ Risk Score")

    st.metric("Risk Level", "18%")

    st.success("Low Risk")

st.divider()

# ------------------ Risk Analysis ------------------
st.subheader("🧠 Explainable AI Analysis")

st.warning("""
Reason for Decision

• Typing speed matches enrolled profile

• Key Hold Duration deviation : 6%

• Flight Time deviation : 4%

• Mouse movement pattern normal

Decision:
Authenticated User
""")

st.divider()

# ------------------ Authentication Log ------------------
st.subheader("📋 Continuous Authentication Log")

st.table({
    "Time":["10:15","10:17","10:19","10:21"],
    "User":["User_01","User_01","User_01","User_01"],
    "Status":["✅ Authenticated",
              "✅ Authenticated",
              "⚠ Slight Variation",
              "✅ Authenticated"]
})
