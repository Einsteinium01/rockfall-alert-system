import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# 1️⃣ Generate Synthetic Data for One Mine
# -------------------------------
def generate_synthetic_data_one_mine(n_samples=100, seed=42):
    np.random.seed(seed)

    # Single mine: Jharia Coalfield (approx coordinates)
    mine_name = "Jharia Coalfield"
    lat_center, lon_center = 23.7426, 86.4111

    slope = np.random.uniform(20, 80, n_samples)
    rainfall = np.random.uniform(0, 200, n_samples)
    displacement = np.random.uniform(0, 10, n_samples)
    vibration = np.random.uniform(0, 5, n_samples)
    temperature = np.random.uniform(-5, 45, n_samples)

    # Add small random jitter around mine location
    lat = lat_center + np.random.uniform(-0.005, 0.005, n_samples)
    lon = lon_center + np.random.uniform(-0.005, 0.005, n_samples)

    # Risk score & label
    risk_score = 0.3*(slope/80) + 0.3*(rainfall/200) + 0.3*(displacement/10) + 0.1*(vibration/5)
    label = (risk_score > 0.6).astype(int)

    df = pd.DataFrame({
        "mine": [mine_name]*n_samples,
        "slope": slope,
        "rainfall": rainfall,
        "displacement": displacement,
        "vibration": vibration,
        "temperature": temperature,
        "lat": lat,
        "lon": lon,
        "label": label
    })
    return df

# Generate training data
df = generate_synthetic_data_one_mine(n_samples=200)

# -------------------------------
# 2️⃣ Train Model
# -------------------------------
X = df.drop(["label", "mine"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    base_score=0.5,
    random_state=42,
    scale_pos_weight=(len(y) - sum(y)) / sum(y)
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# 3️⃣ Streamlit App
# -------------------------------
st.set_page_config(page_title="Rockfall Risk Map - Jharia", layout="wide")
st.title("🪨 Rockfall Risk Prediction - Jharia Coalfield")
st.caption(f"Phase-1 Prototype | Model Accuracy: {accuracy:.2f}")

# Upload CSV or use synthetic
uploaded_file = st.file_uploader("📂 Upload CSV with DEM/Weather/Sensor data", type=["csv"])
if uploaded_file:
    df_app = pd.read_csv(uploaded_file)
else:
    df_app = generate_synthetic_data_one_mine(n_samples=50)  # small demo

st.write("### 📊 Input Data (first 5 rows)")
st.dataframe(df_app.head())

# Predictions
X_app = df_app.drop(columns=[c for c in df_app.columns if c in ["label","mine"]], errors='ignore')
probs = model.predict_proba(X_app)[:,1]
df_app["Risk_Probability"] = probs
df_app["Risk_Level"] = pd.cut(probs, bins=[0,0.3,0.7,1], labels=["Low","Medium","High"])

st.write("### 🔮 Predictions")
st.dataframe(df_app)

# Alerts
alerts = df_app[df_app["Risk_Probability"] > 0.7]
if not alerts.empty:
    st.error(f"⚠️ ALERT: {len(alerts)} zones at HIGH risk!")
else:
    st.success("✅ No high-risk zones detected.")

# -------------------------------
# 4️⃣ Map Visualization (Single Mine)
# -------------------------------
st.write("### 🗺 Map of Rockfall Risk Zones - Jharia Coalfield")
map_data = df_app[["lat", "lon", "Risk_Probability"]].copy()
st.map(map_data)

# -------------------------------
# 5️⃣ Risk Level Distribution
# -------------------------------
st.write("### 📈 Risk Level Distribution")
st.bar_chart(df_app["Risk_Level"].value_counts())

# Download high-risk report
if not alerts.empty:
    st.download_button(
        label="⬇️ Download High-Risk Zones Report",
        data=alerts.to_csv(index=False),
        file_name="high_risk_alerts.csv",
        mime="text/csv"
    )
