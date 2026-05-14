from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import FIGURES_DIR, METRICS_PATH
from src.predict import CLINICAL_DISCLAIMER, predict_readmission

SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "sample_patient.json"


# -----------------------------
# Utility functions
# -----------------------------
def load_sample_patient() -> dict:
    if SAMPLE_PATH.exists():
        return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    return {}


def get_select_index(options: List[Any], value: Any, default_index: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return default_index


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def risk_color(risk_level: str) -> str:
    if "High" in risk_level:
        return "#ef4444"
    if "Medium" in risk_level:
        return "#f59e0b"
    return "#22c55e"


def risk_emoji(risk_level: str) -> str:
    if "High" in risk_level:
        return "🔴"
    if "Medium" in risk_level:
        return "🟠"
    return "🟢"


# -----------------------------
# CSS
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(14, 165, 233, 0.18), transparent 30%),
                radial-gradient(circle at 90% 0%, rgba(34, 197, 94, 0.18), transparent 30%),
                radial-gradient(circle at 50% 100%, rgba(168, 85, 247, 0.16), transparent 30%),
                linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
            color: #f8fafc;
        }

        .main .block-container {
            max-width: 1400px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 70%, #111827 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        .hero {
            position: relative;
            padding: 2.2rem;
            border-radius: 32px;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.78)),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.25), transparent 35%);
            border: 1px solid rgba(148, 163, 184, 0.26);
            box-shadow: 0 28px 90px rgba(0, 0, 0, 0.42);
            overflow: hidden;
            margin-bottom: 1.2rem;
        }

        .hero::before {
            content: "";
            position: absolute;
            inset: -2px;
            background: linear-gradient(90deg, #38bdf8, #22c55e, #eab308, #a855f7);
            opacity: 0.18;
            filter: blur(26px);
            z-index: 0;
        }

        .hero-content {
            position: relative;
            z-index: 1;
        }

        .hero-kicker {
            display: inline-flex;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            color: #bae6fd;
            background: rgba(14, 165, 233, 0.14);
            border: 1px solid rgba(56, 189, 248, 0.28);
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .hero-title {
            font-size: clamp(2.2rem, 5vw, 4.2rem);
            line-height: 1;
            font-weight: 950;
            letter-spacing: -0.06em;
            margin-bottom: 0.8rem;
            background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 30%, #22c55e 65%, #facc15 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            max-width: 920px;
            color: #cbd5e1;
            font-size: 1.05rem;
            line-height: 1.75;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1.4rem;
        }

        .hero-chip {
            padding: 0.85rem;
            border-radius: 18px;
            background: rgba(2, 6, 23, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.16);
            color: #e2e8f0;
            font-size: 0.9rem;
            font-weight: 700;
            text-align: center;
        }

        .glass {
            padding: 1.35rem;
            border-radius: 28px;
            background: rgba(15, 23, 42, 0.74);
            border: 1px solid rgba(148, 163, 184, 0.20);
            box-shadow: 0 22px 55px rgba(0, 0, 0, 0.32);
            backdrop-filter: blur(14px);
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.22rem;
            font-weight: 900;
            color: #f8fafc;
            margin-bottom: 0.85rem;
        }

        .tiny-text {
            color: #94a3b8;
            font-size: 0.88rem;
            line-height: 1.6;
        }

        .risk-card {
            padding: 1.5rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(2, 6, 23, 0.78), rgba(30, 41, 59, 0.72));
            border: 1px solid rgba(148, 163, 184, 0.22);
            text-align: center;
            box-shadow: 0 22px 55px rgba(0, 0, 0, 0.30);
        }

        .risk-prob {
            font-size: clamp(2.5rem, 6vw, 5rem);
            font-weight: 950;
            line-height: 1;
            letter-spacing: -0.06em;
            margin: 0.45rem 0;
        }

        .risk-label {
            font-size: 1.25rem;
            font-weight: 900;
            margin-top: 0.6rem;
        }

        .prediction-pill {
            display: inline-flex;
            padding: 0.7rem 1rem;
            border-radius: 999px;
            font-weight: 850;
            margin-top: 0.8rem;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.24);
            color: #e2e8f0;
        }

        .recommendation {
            padding: 1.1rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.16), rgba(14, 165, 233, 0.12));
            border: 1px solid rgba(34, 197, 94, 0.28);
            color: #dcfce7;
            line-height: 1.65;
            margin-top: 1rem;
        }

        .explanation {
            padding: 1.1rem;
            border-radius: 22px;
            background: rgba(2, 6, 23, 0.54);
            border-left: 4px solid #38bdf8;
            color: #dbeafe;
            line-height: 1.65;
            margin-top: 1rem;
        }

        .warning-box {
            padding: 1rem;
            border-radius: 22px;
            background: rgba(234, 179, 8, 0.12);
            border: 1px solid rgba(234, 179, 8, 0.26);
            color: #fef3c7;
            line-height: 1.6;
        }

        .patient-card {
            padding: 1rem;
            border-radius: 22px;
            background: rgba(2, 6, 23, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.18);
            margin-bottom: 1rem;
        }

        .patient-card b {
            color: #f8fafc;
        }

        .status-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0;
        }

        .status-card {
            padding: 1rem;
            border-radius: 22px;
            background: rgba(2, 6, 23, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.18);
            text-align: center;
        }

        .status-value {
            font-size: 1.65rem;
            font-weight: 950;
            color: #f8fafc;
        }

        .status-label {
            font-size: 0.82rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }

        div[data-testid="stMetric"] {
            padding: 1rem;
            border-radius: 22px;
            background: rgba(2, 6, 23, 0.48);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.20);
        }

        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 900;
        }

        .stButton > button {
            width: 100%;
            height: 3.35rem;
            border-radius: 20px;
            border: 0;
            color: #020617;
            background: linear-gradient(90deg, #38bdf8, #22c55e, #facc15);
            font-weight: 950;
            font-size: 1.05rem;
            box-shadow: 0 16px 42px rgba(34, 197, 94, 0.30);
            transition: all 0.2s ease-in-out;
        }

        .stButton > button:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 20px 55px rgba(56, 189, 248, 0.40);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.65rem 1rem;
            background: rgba(2, 6, 23, 0.44);
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #cbd5e1;
            font-weight: 800;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, rgba(56, 189, 248, 0.22), rgba(34, 197, 94, 0.18));
            color: #f8fafc;
            border: 1px solid rgba(56, 189, 248, 0.35);
        }

        label, .stSelectbox label, .stNumberInput label, .stTextInput label {
            color: #e2e8f0 !important;
            font-weight: 750 !important;
        }

        .footer {
            text-align: center;
            color: #94a3b8;
            margin-top: 1.5rem;
            font-size: 0.9rem;
        }

        @keyframes floatUp {
            0% {
                opacity: 0;
                transform: translateY(18px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes glowPulse {
            0% {
                box-shadow: 0 0 18px rgba(56, 189, 248, 0.18);
            }
            50% {
                box-shadow: 0 0 38px rgba(34, 197, 94, 0.32);
            }
            100% {
                box-shadow: 0 0 18px rgba(56, 189, 248, 0.18);
            }
        }

        @keyframes pulseRisk {
            0% {
                transform: scale(1);
                filter: brightness(1);
            }
            50% {
                transform: scale(1.015);
                filter: brightness(1.15);
            }
            100% {
                transform: scale(1);
                filter: brightness(1);
            }
        }

        .hero {
            animation: floatUp 0.8s ease-out, glowPulse 4s ease-in-out infinite;
        }

        .glass {
            animation: floatUp 0.7s ease-out;
            transition: transform 0.25s ease, border 0.25s ease, box-shadow 0.25s ease;
        }

        .glass:hover {
            transform: translateY(-4px);
            border: 1px solid rgba(56, 189, 248, 0.35);
            box-shadow: 0 26px 70px rgba(14, 165, 233, 0.18);
        }

        .risk-card {
            animation: floatUp 0.7s ease-out;
        }

        .risk-prob {
            animation: pulseRisk 2.2s ease-in-out infinite;
        }

        .hero-chip {
            transition: transform 0.25s ease, background 0.25s ease;
        }

        .hero-chip:hover {
            transform: translateY(-3px) scale(1.03);
            background: rgba(56, 189, 248, 0.16);
        }

        .status-card {
            transition: transform 0.25s ease, border 0.25s ease;
        }

        .status-card:hover {
            transform: translateY(-3px);
            border: 1px solid rgba(34, 197, 94, 0.35);
        }

        .patient-card {
            animation: floatUp 0.7s ease-out;
        }

        @media (max-width: 900px) {
            .hero-grid, .status-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# UI components
# -----------------------------
def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🏥 CareGuard AI")
        st.caption("Hospital Readmission Risk Scorer")

        st.divider()

        st.markdown("### 🚀 System Status")
        st.success("Model loaded")
        st.info("FastAPI ready")
        st.warning("Clinical demo only")

        st.divider()

        st.markdown("### 🧠 MLOps Pipeline")
        st.markdown(
            """
            ✅ Data preprocessing  
            ✅ Missing value handling  
            ✅ Feature engineering  
            ✅ Model comparison  
            ✅ Threshold tuning  
            ✅ MLflow tracking  
            ✅ FastAPI deployment  
            ✅ Streamlit dashboard  
            """
        )

        st.divider()

        st.markdown("### 🔗 Local Demo Links")
        st.code("http://127.0.0.1:8000/docs", language="text")
        st.code("http://127.0.0.1:5000", language="text")

        st.divider()
        st.caption("Built for hackathon demonstration.")


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="hero-kicker">Recall-Focused Clinical MLOps Demo</div>
                <div class="hero-title">CareGuard AI</div>
                <div class="hero-subtitle">
                    A modern hospital intelligence dashboard that estimates 30-day diabetic patient readmission risk,
                    prioritizes recall to reduce missed high-risk cases, and delivers explainable care recommendations.
                </div>
                <div class="hero-grid">
                    <div class="hero-chip">⚕️ Clinical Risk Scoring</div>
                    <div class="hero-chip">🧠 ML Pipeline</div>
                    <div class="hero-chip">📊 MLflow Tracking</div>
                    <div class="hero-chip">🚀 FastAPI + Streamlit</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metrics_panel(metrics: dict) -> None:
    st.markdown('<div class="section-title">📊 Model Performance Snapshot</div>', unsafe_allow_html=True)

    if not metrics:
        st.info("Run `python src/train.py` to generate model metrics.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    c2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    c3.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
    c4.metric("False Negatives", metrics.get("false_negatives", "N/A"))

    st.markdown(
        """
        <div class="explanation">
        <b>Why recall matters:</b> In hospital readmission prediction, a false negative means a truly high-risk patient may not receive timely follow-up care. 
        This project intentionally prioritizes recall over raw accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 View full metrics JSON"):
        st.json(metrics)


def evaluation_graphs() -> None:
    with st.expander("📈 View Model Evaluation Graphs", expanded=False):
        c1, c2, c3 = st.columns(3)

        figures = [
            ("Confusion Matrix", FIGURES_DIR / "confusion_matrix.png"),
            ("ROC Curve", FIGURES_DIR / "roc_curve.png"),
            ("Precision-Recall Curve", FIGURES_DIR / "pr_curve.png"),
        ]

        for col, (title, path) in zip([c1, c2, c3], figures):
            with col:
                st.markdown(f"**{title}**")
                if path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.info("Run training to generate this figure.")


# -----------------------------
# Main app
# -----------------------------
def main() -> None:
    st.set_page_config(
        page_title="CareGuard AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()
    sidebar()
    hero()

    sample = load_sample_patient()
    metrics = load_metrics()

    st.markdown(
        f"""
        <div class="warning-box">
        ⚠️ <b>Clinical Disclaimer:</b> {CLINICAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    left, right = st.columns([1.12, 0.88], gap="large")

    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧾 Patient Intelligence Input</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="tiny-text">Enter patient admission, diagnosis, and diabetes-care information. '
            'The model will estimate readmission probability and generate a care recommendation.</div>',
            unsafe_allow_html=True,
        )

        with st.form("patient_form"):
            tab1, tab2, tab3, tab4 = st.tabs(
                ["👤 Patient", "🏥 Admission", "🧪 Clinical", "💊 Diabetes Care"]
            )

            with tab1:
                patient_name = st.text_input("Patient Name", value="Demo Patient")
                patient_id = st.text_input("Patient ID", value="CG-001")
                care_coordinator = st.text_input("Doctor / Care Coordinator", value="Dr. Demo")

                race_options = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]
                gender_options = ["Female", "Male", "Unknown/Invalid"]
                age_options = [
                    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
                ]

                race = st.selectbox(
                    "Race",
                    race_options,
                    index=get_select_index(race_options, sample.get("race", "Asian"), 2),
                )
                gender = st.selectbox(
                    "Gender",
                    gender_options,
                    index=get_select_index(gender_options, sample.get("gender", "Male"), 1),
                )
                age = st.selectbox(
                    "Age",
                    age_options,
                    index=get_select_index(age_options, sample.get("age", "[40-50)"), 4),
                )

            with tab2:
                c1, c2 = st.columns(2)

                with c1:
                    admission_type_id = st.number_input(
                        "Admission Type ID",
                        min_value=1,
                        max_value=8,
                        value=int(sample.get("admission_type_id", 1)),
                    )
                    admission_source_id = st.number_input(
                        "Admission Source ID",
                        min_value=1,
                        max_value=25,
                        value=int(sample.get("admission_source_id", 7)),
                    )

                with c2:
                    discharge_disposition_id = st.number_input(
                        "Discharge Disposition ID",
                        min_value=1,
                        max_value=30,
                        value=int(sample.get("discharge_disposition_id", 1)),
                    )
                    time_in_hospital = st.number_input(
                        "Time in hospital",
                        min_value=1,
                        max_value=30,
                        value=int(sample.get("time_in_hospital", 2)),
                    )

            with tab3:
                c1, c2 = st.columns(2)

                with c1:
                    num_lab_procedures = st.number_input(
                        "Number of lab procedures",
                        min_value=0,
                        max_value=150,
                        value=int(sample.get("num_lab_procedures", 5)),
                    )
                    num_procedures = st.number_input(
                        "Number of procedures",
                        min_value=0,
                        max_value=20,
                        value=int(sample.get("num_procedures", 1)),
                    )
                    num_medications = st.number_input(
                        "Number of medications",
                        min_value=0,
                        max_value=100,
                        value=int(sample.get("num_medications", 18)),
                    )
                    number_diagnoses = st.number_input(
                        "Number of diagnoses",
                        min_value=1,
                        max_value=30,
                        value=int(sample.get("number_diagnoses", 8)),
                    )

                with c2:
                    number_outpatient = st.number_input(
                        "Previous outpatient visits",
                        min_value=0,
                        max_value=50,
                        value=int(sample.get("number_outpatient", 1)),
                    )
                    number_emergency = st.number_input(
                        "Previous emergency visits",
                        min_value=0,
                        max_value=50,
                        value=int(sample.get("number_emergency", 0)),
                    )
                    number_inpatient = st.number_input(
                        "Previous inpatient visits",
                        min_value=0,
                        max_value=50,
                        value=int(sample.get("number_inpatient", 2)),
                    )

            with tab4:
                c1, c2 = st.columns(2)

                with c1:
                    diag_1 = st.text_input("Diagnosis 1", value=str(sample.get("diag_1", "250.83")))
                    diag_2 = st.text_input("Diagnosis 2", value=str(sample.get("diag_2", "401")))
                    diag_3 = st.text_input("Diagnosis 3", value=str(sample.get("diag_3", "428")))

                    a1c_options = ["None", "Norm", ">7", ">8"]
                    max_glu_options = ["None", "Norm", ">200", ">300"]

                    a1c = st.selectbox(
                        "A1C result",
                        a1c_options,
                        index=get_select_index(a1c_options, sample.get("A1Cresult", ">8"), 3),
                    )
                    max_glu_serum = st.selectbox(
                        "Max glucose serum",
                        max_glu_options,
                        index=get_select_index(max_glu_options, sample.get("max_glu_serum", ">200"), 2),
                    )

                with c2:
                    medicine_options = ["No", "Steady", "Up", "Down"]
                    change_options = ["No", "Ch"]
                    diabetes_med_options = ["No", "Yes"]

                    metformin = st.selectbox(
                        "Metformin",
                        medicine_options,
                        index=get_select_index(medicine_options, sample.get("metformin", "Steady"), 1),
                    )
                    insulin = st.selectbox(
                        "Insulin",
                        medicine_options,
                        index=get_select_index(medicine_options, sample.get("insulin", "Up"), 2),
                    )
                    change = st.selectbox(
                        "Medication change",
                        change_options,
                        index=get_select_index(change_options, sample.get("change", "Ch"), 1),
                    )
                    diabetes_med = st.selectbox(
                        "Diabetes medication",
                        diabetes_med_options,
                        index=get_select_index(diabetes_med_options, sample.get("diabetesMed", "Yes"), 1),
                    )

            submitted = st.form_submit_button("🚀 Generate Readmission Risk Score")

        st.markdown("</div>", unsafe_allow_html=True)

    patient_payload = {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": a1c,
        "metformin": metformin,
        "insulin": insulin,
        "change": change,
        "diabetesMed": diabetes_med,
    }

    display_patient = {
        "patient_name": patient_name,
        "patient_id": patient_id,
        "care_coordinator": care_coordinator,
    }

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 AI Risk Command Center</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="patient-card">
                <b>Patient:</b> {display_patient["patient_name"]}<br>
                <b>Patient ID:</b> {display_patient["patient_id"]}<br>
                <b>Care Coordinator:</b> {display_patient["care_coordinator"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if submitted:
            try:
                with st.spinner("CareGuard AI is analyzing patient risk..."):
                    result = predict_readmission(patient_payload)

                probability = float(result["risk_probability"])
                probability_percent = probability * 100
                color = risk_color(result["risk_level"])
                emoji = risk_emoji(result["risk_level"])

                st.markdown(
                    f"""
                    <div class="risk-card">
                        <div class="tiny-text">Predicted 30-Day Readmission Risk</div>
                        <div class="risk-prob" style="color:{color};">{probability_percent:.1f}%</div>
                        <div class="risk-label" style="color:{color};">{emoji} {result["risk_level"]}</div>
                        <div class="prediction-pill">{result["prediction"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(min(max(probability, 0.0), 1.0))

                st.markdown(
                    f"""
                    <div class="status-row">
                        <div class="status-card">
                            <div class="status-value">{result["threshold_used"]:.2f}</div>
                            <div class="status-label">Threshold</div>
                        </div>
                        <div class="status-card">
                            <div class="status-value">{metrics.get("recall", 0):.2f}</div>
                            <div class="status-label">Recall</div>
                        </div>
                        <div class="status-card">
                            <div class="status-value">{metrics.get("precision", 0):.2f}</div>
                            <div class="status-label">Precision</div>
                        </div>
                        <div class="status-card">
                            <div class="status-value">{metrics.get("false_negatives", 0)}</div>
                            <div class="status-label">False Negatives</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="explanation">
                        <b>🧠 Explanation</b><br>
                        {result["explanation"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="recommendation">
                        <b>✅ Recommended Care Action</b><br>
                        {result["recommendation"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="warning-box">
                        ⚠️ <b>Disclaimer:</b> {result["clinical_disclaimer"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except FileNotFoundError:
                st.error("Model not found. First run `python src/train.py` from the project root.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        else:
            st.markdown(
                """
                <div class="risk-card">
                    <div class="tiny-text">Waiting for patient input</div>
                    <div class="risk-prob" style="color:#38bdf8;">--%</div>
                    <div class="risk-label" style="color:#cbd5e1;">Risk score not generated yet</div>
                    <div class="prediction-pill">Fill the form and generate risk score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="explanation">
                    <b>Demo Tip:</b><br>
                    Use prior inpatient visits, medication change, high A1C, and insulin adjustment to demonstrate a strong readmission-risk scenario.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("🧾 View Patient JSON Sent to Model"):
            st.json(patient_payload)

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    bottom_left, bottom_right = st.columns([1, 1], gap="large")

    with bottom_left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        metrics_panel(metrics)
        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧬 Technical Transparency</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="tiny-text">
            CareGuard AI compares models, logs experiments using MLflow, saves model artifacts,
            tunes thresholds for recall, and exposes predictions through both FastAPI and Streamlit.
            </div>
            """,
            unsafe_allow_html=True,
        )
        evaluation_graphs()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer">
            CareGuard AI • Recall-Focused Hospital Readmission Risk Scorer • Hackathon MLOps Demo
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()