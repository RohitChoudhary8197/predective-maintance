"""Streamlit version of the Engine Predictive Maintenance project."""

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from database import (
    get_predictions,
    get_stats,
    get_user_by_email,
    init_db,
    register_user,
    save_streamlit_prediction,
)


st.set_page_config(page_title="Engine Predictive Maintenance", page_icon="⚙️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "predictive_maintenance_model.pkl"
IMAGES_DIR = BASE_DIR / "static" / "images"
TYPE_MAP = {"Light": 0, "Medium": 1, "Heavy": 2}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def inject_style():
    st.markdown(
        """
        <style>
          .stApp { background: #07111f; color: #e5e7eb; }
          [data-testid="stHeader"] { background: rgba(7,17,31,.92); }
          [data-testid="stSidebar"] { background: #0b1728; }
          .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
          h1, h2, h3 { color: #f8fafc !important; }
          .hero, .panel, .result-card { border: 1px solid rgba(148,163,184,.18); border-radius: 22px; }
          .hero { padding: 3.2rem; background: radial-gradient(circle at 85% 10%, rgba(245,158,11,.22), transparent 30%), linear-gradient(135deg,#101f38,#091321); }
          .eyebrow { color:#fbbf24; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
          .hero h1 { font-size: clamp(2.2rem,5vw,4.1rem); margin:.55rem 0; line-height:1.1; }
          .gold { color:#fbbf24; }.muted { color:#aeb9c8; font-size:1.08rem; }
          .panel { background: linear-gradient(145deg,rgba(20,35,57,.96),rgba(10,20,35,.96)); padding:1.5rem; }
          .stat { background:#10213a; border:1px solid rgba(245,158,11,.13); border-radius:16px; padding:1.25rem; min-height:132px; }
          .stat-label { color:#aeb9c8; margin:0; }.stat-value { color:#f8fafc; font-size:2rem; font-weight:800; margin:.25rem 0; }
          .result-card { padding:2.1rem; text-align:center; animation: rise .65s ease-out; }
          .safe { background:linear-gradient(145deg,rgba(34,197,94,.20),rgba(8,36,28,.78)); border-color:rgba(34,197,94,.45); }
          .warning { background:linear-gradient(145deg,rgba(245,158,11,.22),rgba(54,37,8,.78)); border-color:rgba(245,158,11,.45); }
          .danger { background:linear-gradient(145deg,rgba(239,68,68,.22),rgba(55,13,20,.78)); border-color:rgba(239,68,68,.48); }
          .health-ring { width:142px; height:142px; margin:0 auto 1.3rem; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:2.3rem; font-weight:800; animation: pulse 1.9s ease-in-out infinite; }
          .safe .health-ring { border:5px solid #22c55e; color:#4ade80; }.warning .health-ring { border:5px solid #f59e0b; color:#fbbf24; }.danger .health-ring { border:5px solid #ef4444; color:#f87171; }
          .detail-row { display:flex; justify-content:space-between; gap:1rem; padding:1rem 0; border-bottom:1px solid rgba(255,255,255,.09); color:#b8c3d1; }.detail-row strong { color:#f8fafc; }
          .login-wrap { max-width:490px; margin:3rem auto; }.login-brand { text-align:center; margin-bottom:1.5rem; }
          @keyframes rise { from { opacity:0; transform:translateY(24px); } to { opacity:1; transform:translateY(0); } }
          @keyframes pulse { 50% { transform:scale(1.045); box-shadow:0 0 32px rgba(245,158,11,.23); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note):
    st.markdown(f'<div class="stat"><p class="stat-label">{label}</p><p class="stat-value">{value}</p><small class="gold">{note}</small></div>', unsafe_allow_html=True)


def render_login():
    st.markdown('<div class="login-wrap"><div class="login-brand"><p class="eyebrow">AI powered monitoring</p><h1>⚙️ Engine Health</h1><p class="muted">Sign in to access your predictive maintenance dashboard.</p></div></div>', unsafe_allow_html=True)
    login_tab, register_tab = st.tabs(["Sign In", "Create Account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email address", placeholder="you@company.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
        if submitted:
            user = get_user_by_email(email.strip().lower()) if email else None
            if user and user["password"] == password:
                st.session_state.user_id = user["id"]
                st.session_state.user_name = user["fullname"]
                st.rerun()
            st.error("Invalid email or password. Please try again.")
    with register_tab:
        with st.form("register_form"):
            fullname = st.text_input("Full name", placeholder="Your full name")
            email = st.text_input("Email address", placeholder="you@company.com", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        if submitted:
            if not fullname or not email or not password:
                st.error("Please complete all fields.")
            elif password != confirm:
                st.error("Passwords do not match. Please try again.")
            elif get_user_by_email(email.strip().lower()):
                st.error("Email already registered. Please sign in.")
            elif register_user(fullname.strip(), email.strip().lower(), password):
                st.success("Registration successful. Please sign in.")
            else:
                st.error("Registration failed. Please try again.")


def render_dashboard():
    stats = get_stats()
    hero_left, hero_right = st.columns([1.3, .7], vertical_alignment="center")
    with hero_left:
        st.markdown('<section class="hero"><p class="eyebrow">Next gen maintenance</p><h1>Engine Health <span class="gold">Monitoring and Failure Prediction Using AI/ML</span></h1><p class="muted">Advanced AI/ML-driven predictive analytics for real-time engine monitoring, intelligent failure forecasting, and proactive maintenance planning.</p></section>', unsafe_allow_html=True)
    with hero_right:
        hero_image = IMAGES_DIR / "engine-hero.jpg.jpg"
        if hero_image.exists():
            st.image(str(hero_image), caption="AI-powered engine monitoring", use_container_width=True)
    st.write("")
    st.markdown("## 📊 Analytics Dashboard")
    st.caption("Real-time system overview and engine health status")
    cols = st.columns(4)
    with cols[0]: metric_card("Total Predictions", stats["total"], "↑ Live dashboard")
    with cols[1]: metric_card("Accuracy Rate", "92%", "✓ Verified")
    with cols[2]: metric_card("Risk Alerts", stats["critical"], "Active monitoring")
    with cols[3]: metric_card("Healthy Engines", stats["healthy"], "Good status")
    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel"><h3>📈 Prediction Trends</h3><p class="muted">Recent prediction activity</p></div>', unsafe_allow_html=True)
        records = pd.DataFrame(get_predictions())
        if not records.empty:
            st.line_chart(records.sort_values("id").set_index("id")["failure_prob"])
        else:
            st.info("Make your first prediction to see the trend.")
    with right:
        st.markdown('<div class="panel"><h3>🛡️ Risk Distribution</h3><p class="muted">Current predictions by risk level</p></div>', unsafe_allow_html=True)
        records = pd.DataFrame(get_predictions())
        if not records.empty:
            st.bar_chart(records["risk_level"].value_counts())
        else:
            st.info("Risk distribution will appear here.")


def calculate_prediction(model, machine_type, air_temp, process_temp, rpm, torque, wear):
    row = {"Type": TYPE_MAP[machine_type], "Air temperature [K]": air_temp, "Process temperature [K]": process_temp, "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear, "Temp_Difference": process_temp - air_temp, "Power_Index": rpm * torque, "Torque_RPM_Ratio": torque / rpm}
    features = pd.DataFrame([row])[list(model.feature_names_in_)]
    probability = float(model.predict_proba(features)[0][1])
    failure = int(model.predict(features)[0])
    if probability >= .6:
        risk, message, days, health = "danger", "Immediate maintenance required.", max(3, int(15 * (1 - probability))), max(10, int(100 - probability * 100))
    elif probability >= .3:
        risk, message, days, health = "warning", "Schedule maintenance soon.", max(15, int(40 * (1 - probability))), max(40, int(75 - probability * 50))
    else:
        risk, message, days, health = "safe", "Engine is healthy.", max(60, int(120 * (1 - probability))), max(70, int(95 - probability * 30))
    return {"risk": risk, "message": message, "probability": probability, "failure": failure, "days": days, "health": health, "machine_type": machine_type, "rpm": rpm, "torque": torque, "wear": wear, "time": datetime.now().strftime("%d %b %Y, %I:%M %p")}


def show_result(result):
    labels = {"safe": ("HEALTHY", "SAFE", "🟢"), "warning": ("WARNING", "MAINTENANCE ADVISED", "🟠"), "danger": ("CRITICAL CONDITION", "IMMEDIATE ATTENTION", "🔴")}
    heading, badge, icon = labels[result["risk"]]
    st.markdown("## Prediction Result")
    result_col, details_col = st.columns(2)
    with result_col:
        st.markdown(f'<div class="result-card {result["risk"]}"><div class="health-ring">{result["health"]}%</div><h2>{heading}</h2><p class="muted">{result["message"]}</p><h4>{icon} {badge}</h4></div>', unsafe_allow_html=True)
    with details_col:
        status = "Failure Likely" if result["failure"] else "No Failure"
        st.markdown(f'<div class="panel"><h3 class="gold">📋 Prediction Details</h3><div class="detail-row"><span>Failure Probability</span><strong>{result["probability"] * 100:.1f}%</strong></div><div class="detail-row"><span>Safe Operating Days</span><strong>{result["days"]} days</strong></div><div class="detail-row"><span>Prediction Status</span><strong>{status}</strong></div><div class="detail-row"><span>Engine RPM</span><strong>{result["rpm"]:.0f} RPM</strong></div><div class="detail-row"><span>Torque</span><strong>{result["torque"]:.1f} Nm</strong></div><div class="detail-row"><span>Checked At</span><strong>{result["time"]}</strong></div></div>', unsafe_allow_html=True)


def render_prediction(model):
    st.title("⚙️ Engine Health Check")
    st.caption("Submit engine parameters to receive a detailed health assessment.")
    form_col, image_col = st.columns([1.25, .75], vertical_alignment="center")
    with form_col:
        st.markdown('<div class="panel"><h3 class="gold">Engine Prediction</h3><p class="muted">Enter the current engine sensor readings below.</p></div>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            first, second = st.columns(2)
            with first:
                machine_type = st.selectbox("🏍️ Engine Type", list(TYPE_MAP), index=1)
                wear = st.number_input("🛠️ Tool Wear (min)", min_value=0.0, max_value=253.0, value=50.0, step=1.0)
                air_temp = st.number_input("🌡️ Air Temperature (K)", min_value=295.35, max_value=304.55, value=298.5, step=.1)
            with second:
                process_temp = st.number_input("🔥 Engine Temperature (K)", min_value=305.75, max_value=313.85, value=309.0, step=.1)
                rpm = st.number_input("⚙️ RPM", min_value=1168.0, max_value=2886.0, value=1500.0, step=1.0)
                torque = st.number_input("⚡ Torque (Nm)", min_value=3.8, max_value=76.6, value=44.0, step=.1)
            submitted = st.form_submit_button("🤖 PREDICT ENGINE HEALTH", type="primary", use_container_width=True)
    with image_col:
        prediction_image = IMAGES_DIR / "predictive-engine.jpg"
        if prediction_image.exists():
            st.image(str(prediction_image), caption="AI engine prediction", use_container_width=True)
        st.markdown('<div class="panel"><p class="gold">✓ Real-time analysis</p><p class="gold">✓ Instant risk alerts</p><p class="gold">✓ Maintenance planning</p></div>', unsafe_allow_html=True)
    if submitted:
        if process_temp <= air_temp:
            st.error("Engine temperature must be higher than air temperature.")
            return
        with st.spinner("AI model is analysing engine health..."):
            result = calculate_prediction(model, machine_type, air_temp, process_temp, rpm, torque, wear)
            save_streamlit_prediction(machine_type, result["health"], result["risk"], result["probability"], result["days"], rpm, torque, wear, air_temp, process_temp, result["failure"])
            st.session_state.last_result = result
        st.success("Prediction completed. Open the Result section to view your report.")
        show_result(result)


def render_result():
    st.title("📋 Prediction Result")
    result = st.session_state.get("last_result")
    if not result:
        st.info("Run an engine prediction first; its result will appear here.")
        return
    st.caption("Your latest engine analysis report")
    show_result(result)


def render_history():
    st.title("📜 Prediction History")
    records = pd.DataFrame(get_predictions())
    if records.empty:
        st.info("No prediction records available yet.")
        return
    visible_columns = ["created_at", "engine_type", "health_score", "risk_level", "failure_prob", "safe_days", "rpm", "torque", "wear", "fail_prediction"]
    st.dataframe(records[visible_columns], hide_index=True, use_container_width=True, column_config={"failure_prob": st.column_config.NumberColumn("Failure Probability", format="%.1%"), "health_score": "Health Score", "risk_level": "Risk Level", "safe_days": "Safe Days"})


def render_info(page):
    if page == "About Project":
        st.title("📖 About Project")
        left, right = st.columns([1.15, .85], vertical_alignment="center")
        with left:
            st.markdown('<div class="panel"><h2>Advanced AI Predictive Maintenance</h2><p class="muted">This system uses a Random Forest machine-learning model trained on the AI4I 2020 Predictive Maintenance Dataset. It analyses critical engine parameters and predicts possible failures before they happen, helping improve reliability, reduce downtime and support proactive maintenance.</p><p>✅ 92% prediction accuracy</p><p>✅ Real-time monitoring</p><p>✅ Five critical input features</p><p>✅ Instant maintenance alerts</p></div>', unsafe_allow_html=True)
        with right:
            image = IMAGES_DIR / "ai-dashboard.jpg.jpg"
            if image.exists(): st.image(str(image), caption="AI-driven maintenance insights", use_container_width=True)
    elif page == "Project":
        st.title("🛠️ Project Technology")
        left, right = st.columns([.8, 1.2], vertical_alignment="center")
        with left:
            image = IMAGES_DIR / "project support engineer.jpg"
            if image.exists(): st.image(str(image), caption="Predictive maintenance workflow", use_container_width=True)
        with right:
            st.markdown('<div class="panel"><h2>Technology Stack</h2><p class="muted">A machine-learning web application for early detection of engine failure risk.</p><p class="gold">Python · Streamlit · Scikit-learn · Pandas · SQLite</p><hr><h3>How it works</h3><p>1. Enter current engine measurements.</p><p>2. The trained ML model evaluates the risk.</p><p>3. Receive health score, failure probability and maintenance advice.</p></div>', unsafe_allow_html=True)
    else:
        st.title("📞 Get In Touch")
        st.caption("We would love to hear from you.")
        first, second, third = st.columns(3)
        with first: st.markdown('<div class="panel"><h3>📧 Email</h3><p class="muted">support@rohitchoudhary8197@gmail.com</p></div>', unsafe_allow_html=True)
        with second: st.markdown('<div class="panel"><h3>📱 Phone</h3><p class="muted">+91-7357919798</p></div>', unsafe_allow_html=True)
        with third: st.markdown('<div class="panel"><h3>📍 Location</h3><p class="muted">Jaipur, India</p></div>', unsafe_allow_html=True)
        image = IMAGES_DIR / "contact-support.png"
        if image.exists():
            st.image(str(image), caption="Predictive maintenance support", use_container_width=True)


def main():
    inject_style()
    init_db()
    if not MODEL_PATH.exists():
        st.error("Predictive maintenance model file is missing from the repository.")
        st.stop()
    if "user_id" not in st.session_state:
        render_login()
        return
    model = load_model()
    st.sidebar.title("⚙️ Engine Health")
    st.sidebar.caption(f"Welcome, {st.session_state.user_name}")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Predict Engine", "Result", "History", "About Project", "Project", "Contact"])
    if st.sidebar.button("↪ Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    if page == "Dashboard": render_dashboard()
    elif page == "Predict Engine": render_prediction(model)
    elif page == "Result": render_result()
    elif page == "History": render_history()
    else: render_info(page)
    st.divider()
    st.caption("© 2026 Engine Predictive Maintenance | Built with ❤️ for predictive maintenance")


if __name__ == "__main__":
    main()
