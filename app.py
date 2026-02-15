import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PyPDF2 import PdfReader
import time

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Health Twin AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #007BFF; }
    .risk-card {
        padding: 30px; border-radius: 20px; color: white; text-align: center;
        font-weight: bold; margin-bottom: 20px; box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    .gpt-box {
        background-color: #161b22; padding: 25px; border-radius: 15px;
        border: 1px solid #30363d; line-height: 1.6; color: #e6edf3;
    }
    div.stButton > button {
        width: 100%; border-radius: 25px; height: 3.5em; background: linear-gradient(45deg, #007BFF, #00d4ff);
        color: white; font-weight: bold; border: none; font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE FUNCTIONS ---
def extract_and_score(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = " ".join([p.extract_text() for p in reader.pages]).lower()
        keywords = {"diabetes": 1, "hypertension": 1, "asthma": 1, "cardiac": 2, "stroke": 2, "heart": 2}
        found = [k.title() for k in keywords if k in text]
        score = sum(keywords[k.lower()] for k in found)
        return found, min(score, 5)
    except: return [], 0

def predict_health(vitals, h_score):
    try:
        r_mod = joblib.load('risk_model.pkl')
        d_mod = joblib.load('dept_model.pkl')
        input_df = pd.DataFrame([vitals])
        input_df['History_Score'] = h_score
        cols = ['Age', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'Temperature', 
                'Oxygen_Sat', 'Chest_Pain', 'Shortness_of_Breath', 'Dizziness', 'Vomiting', 'History_Score']
        r_pred = r_mod.predict(input_df[cols])[0]
        d_pred = d_mod.predict(input_df[cols])[0]
        return ["LOW", "MEDIUM", "HIGH"][r_pred], ["General Medicine", "Cardiology", "Neurology", "Emergency"][d_pred]
    except Exception as e: return f"Error", "Error"

def generate_gpt_summary(risk, dept, vitals, history):
    summary = f"### 🤖 AI Clinical Insights\n"
    summary += f"**Assessment:** Patient is currently classified as **{risk} RISK**. "
    if risk == "HIGH":
        summary += "Immediate clinical intervention is prioritized due to unstable physiological markers. "
    
    summary += "\n\n**Clinical Observations:**\n"
    if vitals['Systolic_BP'] > 140: summary += f"- Hypertensive urgency noted: {vitals['Systolic_BP']} mmHg.\n"
    if vitals['Oxygen_Sat'] < 95: summary += f"- Hypoxia risk: SpO2 levels at {vitals['Oxygen_Sat']}%.\n"
    if history: summary += f"- Relevant Comorbidities: {', '.join(history)}.\n"
    
    summary += f"\n**Recommended Plan:**\n- Transfer to **{dept}**\n- Initiate continuous vitals monitoring\n- Review full EHR for medication contraindications."
    return summary

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843118.png", width=80)
    st.title("Patient Triage")
    file = st.file_uploader("Upload Medical PDF", type=['pdf'])
    h_list, h_score = (extract_and_score(file)) if file else ([], 0)
    
    st.divider()
    age = st.number_input("Age", 1, 100, 45)
    sbp = st.slider("Systolic BP", 80, 200, 120)
    dbp = st.slider("Diastolic BP", 50, 120, 80)
    hr = st.slider("Heart Rate", 40, 160, 75)
    o2 = st.slider("O2 Saturation", 80, 100, 98)
    temp = st.number_input("Temp °F", 95.0, 106.0, 98.6)
    cp, sob = st.checkbox("Chest Pain"), st.checkbox("Shortness of Breath")
    diz, vom = st.checkbox("Dizziness"), st.checkbox("Vomiting")

# --- 4. MAIN DASHBOARD ---
# Patient Header
head1, head2 = st.columns([0.15, 0.85])
with head1: st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
with head2:
    st.title("Health Twin AI: Clinical Dashboard")
    st.markdown(f"**Patient ID:** #BT-83921  |  **Age:** {age}  |  **Status:** Active Analysis")

# Vitals Ribbon
m1, m2, m3, m4 = st.columns(4)
m1.metric("BP Index", f"{sbp}/{dbp}", delta="Normal" if sbp < 130 else "High", delta_color="inverse")
m2.metric("Heart Rate", f"{hr} BPM")
m3.metric("O2 Level", f"{o2}%")
m4.metric("Body Temp", f"{temp}°F")

st.markdown("### ⚡ Live Vitals Stream")
t = np.linspace(time.time(), time.time() + 10, 50)
wave = (np.sin(t) * 2) + (np.random.randn(50) * 0.2)
st.line_chart(pd.DataFrame(wave, columns=['ECG (mV)']), height=150)

st.divider()

# Prediction Logic with Unique Key
if st.button("🚀 INITIATE AI DIAGNOSTICS", key="diagnostics_btn_main"):
    vitals_data = {'Age': age, 'Systolic_BP': sbp, 'Diastolic_BP': dbp, 'Heart_Rate': hr, 'Temperature': temp, 
                   'Oxygen_Sat': o2, 'Chest_Pain': 1 if cp else 0, 'Shortness_of_Breath': 1 if sob else 0, 
                   'Dizziness': 1 if diz else 0, 'Vomiting': 1 if vom else 0}
    
    with st.spinner("AI Generating Recommendation..."):
        time.sleep(1.5)
        risk, dept = predict_health(vitals_data, h_score)
        gpt_rec = generate_gpt_summary(risk, dept, vitals_data, h_list)
    
    col_l, col_r = st.columns([1, 1.2])
    with col_l:
        color = "#2ecc71" if risk == "LOW" else "#f1c40f" if risk == "MEDIUM" else "#e74c3c"
        st.markdown(f'<div class="risk-card" style="background:{color};"><h1>{risk} RISK</h1><p>Triage: {dept}</p></div>', unsafe_allow_html=True)
    
    with col_r:
        st.markdown(f'<div class="gpt-box">{gpt_rec}</div>', unsafe_allow_html=True)
        st.download_button("📥 Export Clinical Report", gpt_rec, file_name="HealthTwin_Report.txt")

st.caption("v2.2 | Clinical Neural Node | Secure Session Active")
# --- 5. HOSPITAL DATASET ---
HOSPITAL_DATABASE = {
    "Cardiology": [
        {"name": "City Heart Institute", "dist": "1.2 miles", "address": "101 Cardiac Way"},
        {"name": "St. Jude’s Vascular Center", "dist": "3.5 miles", "address": "44 Heart St."}
    ],
    "Neurology": [
        {"name": "Neuro-Link Specialty Hospital", "dist": "0.8 miles", "address": "22 Brain Ave."},
        {"name": "Cerebral Health Clinic", "dist": "4.1 miles", "address": "900 Grey Matter Dr."}
    ],
    "Emergency": [
        {"name": "General Trauma Center", "dist": "0.5 miles", "address": "1 Emergency Rd. (24/7)"},
        {"name": "Metro Urgent Care", "dist": "2.2 miles", "address": "55 Rapid Response Ln."}
    ],
    "General Medicine": [
        {"name": "Community Health Plaza", "dist": "1.1 miles", "address": "77 Wellness Blvd."},
        {"name": "Primary Care Partners", "dist": "2.9 miles", "address": "31 Family Row"}
    ]
}

# Initialize Navigation State
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# --- 6. CONDITIONAL PAGE RENDERING ---

# PAGE 1: THE DASHBOARD
if st.session_state.page == "dashboard":
    # ... (Your existing code for metrics, ECG, and diagnostics button goes here) ...
    # Assuming 'dept' is already predicted:
    
    if st.button("🚀 INITIATE AI DIAGNOSTICS", key="main_diag"):
        # [Existing prediction logic here]
        st.session_state.last_dept = "General Medicine" # Placeholder for example
        st.session_state.last_risk = "LOW"
        # (Save actual results to session_state so they persist when switching pages)
    
    st.divider()
    if st.button("📍 Find Nearest Treatment Center", use_container_width=True):
        st.session_state.page = "hospital_finder"
        st.rerun()

# PAGE 2: THE HOSPITAL FINDER
elif st.session_state.page == "hospital_finder":
    st.title("📍 Local Treatment Suggestions")
    target_dept = st.session_state.get('last_dept', 'General Medicine')
    
    st.write(f"Showing best facilities for: **{target_dept}**")
    # 
    for hosp in HOSPITAL_DATABASE.get(target_dept, []):
        with st.container():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"### {hosp['name']}")
                st.caption(f"📍 {hosp['address']}")
            with col_b:
                st.button(f"Navigate ({hosp['dist']})", key=hosp['name'])
            st.divider()

    if st.button("⬅️ Back to Clinical Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()