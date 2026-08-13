import os
import joblib
import pandas as pd
import streamlit as st

# ==========================

# Page Configuration

# ==========================

st.set_page_config(
page_title="AI Predictive Maintenance",
page_icon="⚙️",
layout="wide"
)

# ==========================

# Load Model

# ==========================

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
BASE,
"predictive_maintenance_model.pkl"
)

MODEL = joblib.load(MODEL_PATH)

TYPE_MAP = {
"Light": 0,
"Medium": 1,
"Heavy": 2
}

# ==========================

# Custom CSS

# ==========================

st.markdown(
""" <style>
.main-title {
font-size: 42px;
font-weight: 700;
text-align: center;
margin-bottom: 5px;
}

```
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }

    .danger {
        background-color: #ffe5e5;
        border: 1px solid #ff4d4d;
    }

    .safe {
        background-color: #e5ffe9;
        border: 1px solid #28a745;
    }
</style>
""",
unsafe_allow_html=True
```

)

# ==========================

# Header

# ==========================

st.markdown(
'<div class="main-title">⚙️ AI Predictive Maintenance</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">'
'Predict machine failure risk using Machine Learning'
'</div>',
unsafe_allow_html=True
)

st.divider()

# ==========================

# Input Section

# ==========================

st.subheader("🔧 Machine Parameters")

col1, col2 = st.columns(2)

with col1:
machine_type = st.selectbox(
"Machine Type",
["Light", "Medium", "Heavy"]
)

```
air_temp = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=298.0,
    step=0.1
)

process_temp = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=450.0,
    value=308.0,
    step=0.1
)

rpm = st.number_input(
    "Rotational Speed [rpm]",
    min_value=500.0,
    max_value=5000.0,
    value=1500.0,
    step=10.0
)
```

with col2:
torque = st.number_input(
"Torque [Nm]",
min_value=0.0,
max_value=100.0,
value=40.0,
step=0.1
)

```
tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0.0,
    max_value=300.0,
    value=100.0,
    step=1.0
)
```

st.divider()

# ==========================

# Prediction

# ==========================

if st.button(
"🔍 Predict Machine Health",
type="primary",
use_container_width=True
):

```
if rpm <= 0:
    st.error("Rotational speed must be greater than 0.")
    st.stop()

# Derived features
temp_difference = process_temp - air_temp

power_index = torque * rpm

torque_rpm_ratio = torque / rpm

# Create input data
row = {
    "Type": TYPE_MAP[machine_type],
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rpm,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Temp_Difference": temp_difference,
    "Power_Index": power_index,
    "Torque_RPM_Ratio": torque_rpm_ratio
}

df = pd.DataFrame([row])

# Match exact model feature order
df = df[list(MODEL.feature_names_in_)]

# Model prediction
probability = float(
    MODEL.predict_proba(df)[0][1]
)

failure_prediction = int(
    MODEL.predict(df)[0]
)

# ==========================
# Risk Calculation
# ==========================

if probability >= 0.6:
    risk = "HIGH RISK"
    message = "Immediate maintenance required."

    days = max(
        3,
        int(15 * (1 - probability))
    )

    health = max(
        10,
        int(100 - probability * 100)
    )

else:
    risk = "LOW RISK"
    message = "Machine is operating normally."

    days = max(
        7,
        int(30 * (1 - probability))
    )

    health = min(
        100,
        int(100 - probability * 50)
    )

# ==========================
# Result Summary
# ==========================

st.subheader("📊 Prediction Result")

result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:
    st.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%"
    )

with result_col2:
    st.metric(
        "Machine Health",
        f"{health}%"
    )

with result_col3:
    st.metric(
        "Maintenance Estimate",
        f"{days} days"
    )

if risk == "HIGH RISK":
    st.markdown(
        f"""
        <div class="result-box danger">
            <h2>⚠️ {risk}</h2>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.markdown(
        f"""
        <div class="result-box safe">
            <h2>✅ {risk}</h2>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================
# Detailed Result
# ==========================

st.subheader("🔎 Prediction Details")

result_data = pd.DataFrame(
    {
        "Parameter": [
            "Machine Type",
            "Air Temperature",
            "Process Temperature",
            "Temperature Difference",
            "Rotational Speed",
            "Torque",
            "Tool Wear",
            "Power Index",
            "Torque/RPM Ratio",
            "Failure Prediction"
        ],
        "Value": [
            machine_type,
            f"{air_temp:.2f} K",
            f"{process_temp:.2f} K",
            f"{temp_difference:.2f} K",
            f"{rpm:.0f} rpm",
            f"{torque:.2f} Nm",
            f"{tool_wear:.0f} min",
            f"{power_index:.2f}",
            f"{torque_rpm_ratio:.6f}",
            (
                "Failure Detected"
                if failure_prediction == 1
                else "No Failure"
            )
        ]
    }
)

st.dataframe(
    result_data,
    use_container_width=True,
    hide_index=True
)
```

# ==========================

# Footer

# ==========================

st.divider()

st.caption(
"AI Predictive Maintenance System | "
"Machine Learning Based Failure Prediction"
)

