import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("rockfall_model.pkl", "rb"))

st.title("🪨 Rockfall Risk Prediction - Prototype (Phase 1)")

# Upload CSV or use default
uploaded_file = st.file_uploader("Upload DEM/Weather/Sensor data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("synthetic_data.csv")

st.write("### Input Data")
st.dataframe(df.head())

# Predictions
X = df.drop("label", axis=1)
probs = model.predict_proba(X)[:,1]
df["Risk_Probability"] = probs
df["Risk_Level"] = pd.cut(probs, bins=[0,0.3,0.7,1], labels=["Low","Medium","High"])

st.write("### Predictions")
st.dataframe(df[["slope","rainfall","displacement","Risk_Probability","Risk_Level"]])

# Alerts
alerts = df[df["Risk_Probability"] > 0.7]
if not alerts.empty:
    st.error(f"⚠️ ALERT: {len(alerts)} zones at HIGH risk!")
else:
    st.success("✅ No high-risk zones detected.")
