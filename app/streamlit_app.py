from __future__ import annotations

import json
import sys
import time
from html import escape
from pathlib import Path
from textwrap import dedent
from typing import Any, List, Tuple

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, METRICS_PATH
from src.predict import CLINICAL_DISCLAIMER, predict_readmission

SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "sample_patient.json"


def render_html(markup: str) -> None:
    cleaned = " ".join(dedent(markup).strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def html_string(markup: str) -> str:
    return " ".join(dedent(markup).strip().splitlines())


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


def get_dashboard_risk_level(probability_percent: float) -> Tuple[str, str]:
    if probability_percent <= 30:
        return "Low Risk", "#22c55e"
    if probability_percent <= 60:
        return "Medium Risk", "#f59e0b"
    return "High Risk", "#fb5a1e"


def risk_emoji() -> str:
    return "●"


def risk_dashboard_card(
    percent_value: int | str,
    color: str,
    risk_level: str | None = None,
    prediction: str | None = None,
    show_details: bool = False,
) -> str:
    details_html = ""

    if show_details and risk_level and prediction:
        details_html = f"""
            <div class="risk-divider"></div>

            <div class="risk-dashboard-row">
                <div class="risk-dashboard-label">Risk Level</div>
                <div class="risk-level-badge" style="background:{color};">
                    {risk_emoji()} {escape(str(risk_level))}
                </div>
            </div>

            <div class="risk-dashboard-row">
                <div class="risk-dashboard-label">Prediction</div>
                <div class="risk-dashboard-value">{escape(str(prediction))}</div>
            </div>

            <div class="clinical-note">
                AI-assisted decision support. Final clinical decision must be made by healthcare professionals.
            </div>
        """

    return html_string(
        f"""
        <div class="hospital-risk-output">
            <div class="risk-hero-row">
                <div class="risk-hero-left">
                    <div class="risk-main-label">Readmission Risk</div>
                    <div class="risk-main-sub">
                        {'AI-generated hospital risk score' if show_details else 'Waiting for prediction'}
                    </div>
                </div>
                <div class="risk-percent-display">
                    <div class="risk-number" style="color:{color};">{percent_value}</div>
                    <div class="risk-symbol" style="color:{color};">%</div>
                </div>
            </div>
            {details_html}
        </div>
        """
    )


def animate_risk_score(
    placeholder,
    target_percent: float,
    color: str,
    risk_level: str,
    prediction: str,
) -> None:
    target_int = int(round(target_percent))
    step = max(1, target_int // 34)

    for value in range(0, target_int + 1, step):
        placeholder.markdown(
            risk_dashboard_card(
                percent_value=value,
                color=color,
                risk_level=risk_level,
                prediction=prediction,
                show_details=True,
            ),
            unsafe_allow_html=True,
        )
        time.sleep(0.018)

    placeholder.markdown(
        risk_dashboard_card(
            percent_value=target_int,
            color=color,
            risk_level=risk_level,
            prediction=prediction,
            show_details=True,
        ),
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    render_html(
        """
        <style>
        :root {
            --main-bg:#0B0B0B;
            --secondary-bg:#111111;
            --card-bg:rgba(24,24,24,0.92);
            --deeper-panel:#050505;
            --accent:#FC8019;
            --accent-bright:#FF9F45;
            --accent-dark:#D96B12;
            --accent-glow:rgba(252,128,25,0.35);
            --text:#FFFFFF;
            --text-secondary:#D1D5DB;
            --muted:#9CA3AF;
            --border:rgba(255,255,255,0.10);
            --orange-border:rgba(252,128,25,0.25);
            --green:#22c55e;
            --amber:#f59e0b;
            --orange:#fb5a1e;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(252,128,25,0.20), transparent 28%),
                radial-gradient(circle at 84% 14%, rgba(255,159,69,0.12), transparent 30%),
                radial-gradient(circle at 50% 100%, rgba(252,128,25,0.09), transparent 38%),
                linear-gradient(135deg, #0B0B0B 0%, #050505 45%, #111111 100%) !important;
            background-size: 140% 140%;
            color: var(--text);
            animation: appAuraShift 18s ease-in-out infinite alternate;
        }

        header[data-testid="stHeader"] {
            background: rgba(11,11,11,0.88);
            backdrop-filter: blur(18px);
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }

        .main .block-container {
            max-width: 1420px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            animation: dashboardEnter 650ms ease-out both;
        }

        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 15% 0%, rgba(252,128,25,0.18), transparent 30%),
                linear-gradient(180deg, rgba(11,11,11,0.98) 0%, rgba(5,5,5,0.98) 52%, rgba(17,17,17,0.98) 100%);
            border-right: 1px solid rgba(252,128,25,0.24);
            box-shadow: 18px 0 60px rgba(0,0,0,0.50);
        }

        section[data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        [data-testid="stAlert"] {
            background: rgba(24,24,24,0.82) !important;
            border: 1px solid rgba(252,128,25,0.18) !important;
            border-radius: 18px !important;
            color: var(--text) !important;
            box-shadow: 0 12px 34px rgba(0,0,0,0.26);
        }

        [data-testid="stAlert"] * {
            color: var(--text) !important;
        }

        .hero {
            position: relative;
            padding: 2.35rem;
            border-radius: 34px;
            background:
                linear-gradient(135deg, rgba(24,24,24,0.94), rgba(5,5,5,0.86)),
                radial-gradient(circle at top right, rgba(252,128,25,0.18), transparent 35%),
                radial-gradient(circle at bottom left, rgba(255,159,69,0.11), transparent 35%);
            border: 1px solid var(--orange-border);
            box-shadow:
                0 34px 110px rgba(0,0,0,0.56),
                0 0 56px rgba(252,128,25,0.10),
                inset 0 1px 0 rgba(255,255,255,0.08);
            overflow: hidden;
            margin-bottom: 1.2rem;
            animation: softLift 700ms ease-out both;
        }

        .hero::before {
            content: "";
            position: absolute;
            inset: -2px;
            background: linear-gradient(90deg, rgba(252,128,25,0.22), rgba(255,159,69,0.14), rgba(255,255,255,0.05));
            opacity: 0.55;
            filter: blur(32px);
            z-index: 0;
            animation: premiumGlow 5s ease-in-out infinite;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(105deg, transparent 0%, transparent 42%, rgba(255,255,255,0.055) 50%, transparent 58%, transparent 100%);
            transform: translateX(-120%);
            animation: glassSweep 7s ease-in-out infinite;
            pointer-events: none;
        }

        .hero-content {
            position: relative;
            z-index: 1;
        }

        .hero-kicker {
            display: inline-flex;
            padding: 0.48rem 0.9rem;
            border-radius: 999px;
            color: #fff7ed;
            background: rgba(252,128,25,0.13);
            border: 1px solid rgba(252,128,25,0.34);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.055em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
            box-shadow: 0 0 26px rgba(252,128,25,0.16);
        }

        .hero-title {
            font-size: clamp(2.4rem, 5vw, 4.45rem);
            line-height: 1;
            font-weight: 950;
            letter-spacing: -0.065em;
            margin-bottom: 0.9rem;
            background: linear-gradient(90deg, #FFFFFF 0%, #fff7ed 32%, #FF9F45 64%, #FC8019 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 12px 34px rgba(252,128,25,0.13));
        }

        .hero-subtitle {
            max-width: 960px;
            color: var(--text-secondary);
            font-size: 1.06rem;
            line-height: 1.76;
            font-weight: 500;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1.45rem;
        }

        .hero-chip {
            padding: 0.9rem;
            border-radius: 20px;
            background: rgba(24,24,24,0.82);
            border: 1px solid var(--border);
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 850;
            text-align: center;
            box-shadow: 0 18px 45px rgba(0,0,0,0.34);
            transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
            animation: chipFloat 4.6s ease-in-out infinite;
        }

        .hero-chip:nth-child(2) { animation-delay: 0.3s; }
        .hero-chip:nth-child(3) { animation-delay: 0.6s; }
        .hero-chip:nth-child(4) { animation-delay: 0.9s; }

        .hero-chip:hover {
            transform: translateY(-4px);
            border-color: rgba(255,159,69,0.44);
            box-shadow: 0 26px 70px rgba(0,0,0,0.44), 0 0 36px rgba(252,128,25,0.15);
        }

        div[data-testid="stForm"] {
            position: relative;
            overflow: hidden;
            padding: 1.2rem;
            border-radius: 30px;
            background: rgba(24,24,24,0.84);
            border: 1px solid var(--orange-border);
            box-shadow: 0 26px 78px rgba(0,0,0,0.46);
            animation: softLift 600ms ease-out both;
        }

        div[data-testid="stForm"]::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(105deg, transparent 0%, transparent 44%, rgba(255,159,69,0.065) 50%, transparent 56%, transparent 100%);
            transform: translateX(-120%);
            animation: formSweep 9s ease-in-out infinite;
            pointer-events: none;
        }

        .section-title {
            position: relative;
            display: inline-block;
            font-size: 1.24rem;
            font-weight: 950;
            color: var(--text);
            margin-bottom: 0.85rem;
            animation: softLift 650ms ease-out both;
        }

        .section-title::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: -8px;
            width: 54%;
            height: 2px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent-bright), var(--accent), transparent);
            box-shadow: 0 0 20px rgba(252,128,25,0.42);
            animation: underlinePulse 3.2s ease-in-out infinite;
        }

        .tiny-text {
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.65;
        }

        .hospital-risk-output,
        .patient-card,
        .status-card,
        div[data-testid="stMetric"] {
            transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
        }

        .hospital-risk-output:hover,
        .patient-card:hover,
        .status-card:hover,
        div[data-testid="stMetric"]:hover {
            transform: translateY(-4px);
            border-color: rgba(255,159,69,0.42);
            box-shadow: 0 30px 90px rgba(0,0,0,0.52), 0 0 44px rgba(252,128,25,0.14);
        }

        .hospital-risk-output {
            position: relative;
            overflow: hidden;
            padding: 1.45rem;
            border-radius: 30px;
            background:
                linear-gradient(135deg, rgba(24,24,24,0.96), rgba(5,5,5,0.90)),
                radial-gradient(circle at top right, rgba(252,128,25,0.12), transparent 44%);
            border: 1px solid var(--orange-border);
            box-shadow: 0 30px 90px rgba(0,0,0,0.52), 0 0 42px rgba(252,128,25,0.10);
            margin-top: 1rem;
            animation: softLift 700ms ease-out both;
        }

        .hospital-risk-output::before {
            content: "";
            position: absolute;
            inset: -40%;
            background: conic-gradient(from 180deg, transparent, rgba(255,159,69,0.12), transparent, rgba(252,128,25,0.08), transparent);
            animation: cardAuraRotate 8s linear infinite;
            pointer-events: none;
        }

        .hospital-risk-output > * {
            position: relative;
            z-index: 1;
        }

        .risk-hero-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
        }

        .risk-hero-left {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .risk-main-label {
            color: #FFFFFF;
            font-size: 1rem;
            font-weight: 850;
            letter-spacing: 0.01em;
        }

        .risk-main-sub {
            color: #D1D5DB;
            font-size: 0.82rem;
            font-weight: 700;
            opacity: 0.94;
        }

        .risk-percent-display {
            display: flex;
            align-items: flex-start;
            justify-content: flex-end;
            gap: 0.15rem;
            line-height: 0.9;
        }

        .risk-number {
            font-size: clamp(3.4rem, 6.5vw, 5.4rem);
            font-weight: 950;
            letter-spacing: -0.06em;
            text-shadow: 0 0 26px rgba(252,128,25,0.20);
            min-width: 2.4ch;
            text-align: right;
        }

        .risk-symbol {
            font-size: clamp(2rem, 3.3vw, 3rem);
            font-weight: 950;
            line-height: 1;
            margin-top: 0.35rem;
            opacity: 0.98;
            text-shadow: 0 0 18px rgba(252,128,25,0.22);
        }

        .risk-divider {
            width: 100%;
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(252,128,25,0.26), rgba(255,255,255,0.08));
            margin: 1rem 0 1.1rem 0;
        }

        .risk-dashboard-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            padding: 0.85rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }

        .risk-dashboard-row:last-child {
            border-bottom: none;
        }

        .risk-dashboard-label {
            color: #D1D5DB;
            font-weight: 850;
            font-size: 0.98rem;
        }

        .risk-dashboard-value {
            color: #FFFFFF;
            font-weight: 950;
            font-size: 1.2rem;
            text-align: right;
        }

        .risk-level-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.68rem 1.2rem;
            border-radius: 999px;
            color: #050505;
            font-weight: 950;
            font-size: 1rem;
            box-shadow: 0 0 30px rgba(255,255,255,0.08);
            min-width: 180px;
        }

        .clinical-note {
            margin-top: 1.1rem;
            padding: 1rem 1.05rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(252,128,25,0.10), rgba(24,24,24,0.70));
            border: 1px solid rgba(252,128,25,0.24);
            color: #fff7ed;
            font-size: 0.96rem;
            line-height: 1.7;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }

        .recommendation {
            padding: 1.1rem;
            border-radius: 24px;
            background: rgba(34,197,94,0.10);
            border: 1px solid rgba(34,197,94,0.22);
            color: #dcfce7;
            line-height: 1.65;
            margin-top: 1rem;
            animation: softLift 700ms ease-out both;
        }

        .explanation {
            padding: 1.1rem;
            border-radius: 24px;
            background: rgba(24,24,24,0.86);
            border-left: 4px solid var(--accent);
            border-top: 1px solid rgba(255,255,255,0.08);
            border-right: 1px solid rgba(255,255,255,0.08);
            border-bottom: 1px solid rgba(255,255,255,0.08);
            color: var(--text-secondary);
            line-height: 1.65;
            margin-top: 1rem;
            animation: softLift 700ms ease-out both;
        }

        .warning-box {
            padding: 1rem;
            border-radius: 24px;
            background: rgba(252,128,25,0.10);
            border: 1px solid rgba(252,128,25,0.25);
            color: #FFFFFF;
            line-height: 1.6;
            animation: softLift 700ms ease-out both;
            box-shadow: 0 16px 44px rgba(0,0,0,0.32);
        }

        .patient-card {
            padding: 1rem;
            border-radius: 24px;
            background: rgba(24,24,24,0.86);
            border: 1px solid var(--orange-border);
            margin-bottom: 1rem;
            color: var(--text-secondary);
            animation: softLift 650ms ease-out both;
            box-shadow: 0 18px 54px rgba(0,0,0,0.38);
        }

        .patient-card b {
            color: var(--text);
        }

        .status-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 1rem 0;
        }

        .status-card {
            padding: 1rem;
            border-radius: 24px;
            background: rgba(24,24,24,0.86);
            border: 1px solid var(--border);
            text-align: center;
            animation: softLift 700ms ease-out both;
            box-shadow: 0 18px 54px rgba(0,0,0,0.34);
        }

        .status-value {
            font-size: 1.65rem;
            font-weight: 950;
            color: var(--text);
        }

        .status-label {
            font-size: 0.82rem;
            color: var(--muted);
            margin-top: 0.25rem;
        }

        div[data-testid="stMetric"] {
            padding: 1rem;
            border-radius: 24px;
            background: rgba(24,24,24,0.86);
            border: 1px solid var(--border);
            box-shadow: 0 18px 50px rgba(0,0,0,0.34);
            animation: softLift 700ms ease-out both;
        }

        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 950;
            color: var(--text);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--text-secondary);
            font-weight: 800;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            width: 100%;
            min-height: 3.35rem;
            border-radius: 22px;
            border: 0;
            color: #050505 !important;
            background: linear-gradient(90deg, #FF9F45, #FC8019, #D96B12) !important;
            background-size: 220% 220% !important;
            font-weight: 950;
            font-size: 1.05rem;
            box-shadow: 0 18px 55px rgba(252,128,25,0.24), 0 0 0 1px rgba(255,255,255,0.12) inset;
            transition: all 0.25s ease-in-out;
            animation: buttonGradientFlow 4.8s ease-in-out infinite;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            transform: translateY(-3px) scale(1.018);
            box-shadow: 0 30px 84px rgba(252,128,25,0.36), 0 0 50px rgba(255,159,69,0.22), 0 0 0 1px rgba(255,255,255,0.22) inset;
            filter: brightness(1.08);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.65rem 1rem;
            background: rgba(24,24,24,0.84);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            font-weight: 850;
            transition: all 220ms ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            border-color: rgba(252,128,25,0.34);
            transform: translateY(-1px);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(252,128,25,0.16);
            color: var(--text);
            border: 1px solid rgba(252,128,25,0.40);
            box-shadow: 0 0 28px rgba(252,128,25,0.14);
        }

        label,
        .stSelectbox label,
        .stNumberInput label,
        .stTextInput label {
            color: var(--text) !important;
            font-weight: 850 !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] div[data-baseweb="input"] input,
        [data-testid="stNumberInput"] div[data-baseweb="input"] input {
            background-color: transparent !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            caret-color: #FF9F45 !important;
        }

        [data-testid="stTextInput"] div[data-baseweb="input"],
        [data-testid="stNumberInput"] div[data-baseweb="input"],
        [data-testid="stTextInput"] div[data-baseweb="base-input"],
        [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
            background-color: rgba(5,5,5,0.94) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 17px !important;
            color: #FFFFFF !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 12px 32px rgba(0,0,0,0.34) !important;
            transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
        }

        [data-testid="stTextInput"] div[data-baseweb="input"] > div,
        [data-testid="stNumberInput"] div[data-baseweb="input"] > div {
            background-color: transparent !important;
        }

        input::placeholder {
            color: rgba(209,213,219,0.58) !important;
            -webkit-text-fill-color: rgba(209,213,219,0.58) !important;
        }

        [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
        [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
            transform: translateY(-1px);
            border-color: rgba(255,159,69,0.58) !important;
            box-shadow: 0 0 0 1px rgba(252,128,25,0.22), 0 0 34px rgba(252,128,25,0.16) !important;
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: rgba(5,5,5,0.94) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 17px !important;
            color: #FFFFFF !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 12px 32px rgba(0,0,0,0.34) !important;
            transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
            transform: translateY(-1px);
            border-color: rgba(255,159,69,0.46) !important;
            box-shadow: 0 0 0 1px rgba(252,128,25,0.15), 0 0 30px rgba(252,128,25,0.12) !important;
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] span {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        [data-testid="stSelectbox"] svg {
            fill: #FFFFFF !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] ul,
        div[role="listbox"] {
            background: #111111 !important;
            color: var(--text) !important;
            border-radius: 18px !important;
            border: 1px solid rgba(252,128,25,0.24);
            box-shadow: 0 18px 60px rgba(0,0,0,0.55);
        }

        div[role="option"] {
            background: #111111 !important;
            color: var(--text) !important;
        }

        div[role="option"]:hover {
            background: rgba(252,128,25,0.18) !important;
            color: var(--text) !important;
        }

        input, textarea {
            color: var(--text) !important;
        }

        div[data-testid="stExpander"] {
            background: rgba(24,24,24,0.76) !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            border-radius: 20px !important;
            box-shadow: 0 16px 48px rgba(0,0,0,0.28);
            overflow: hidden;
        }

        div[data-testid="stExpander"] * {
            color: var(--text-secondary) !important;
        }

        pre, code, div[data-testid="stJson"] {
            background: rgba(5,5,5,0.96) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(252,128,25,0.18) !important;
            border-radius: 16px !important;
        }

        .footer {
            text-align: center;
            color: var(--muted);
            margin-top: 1.5rem;
            font-size: 0.9rem;
            font-weight: 650;
            animation: softLift 900ms ease-out both;
        }

        @keyframes appAuraShift {
            0% { background-position: 0% 45%; }
            100% { background-position: 100% 55%; }
        }

        @keyframes softLift {
            0% { opacity: 0; transform: translateY(12px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        @keyframes dashboardEnter {
            0% { opacity: 0; transform: translateY(8px); filter: blur(3px); }
            100% { opacity: 1; transform: translateY(0); filter: blur(0); }
        }

        @keyframes premiumGlow {
            0%, 100% { opacity: 0.38; }
            50% { opacity: 0.78; }
        }

        @keyframes glassSweep {
            0%, 62% { transform: translateX(-120%); }
            100% { transform: translateX(120%); }
        }

        @keyframes formSweep {
            0%, 72% { transform: translateX(-120%); }
            100% { transform: translateX(120%); }
        }

        @keyframes chipFloat {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-2px); }
        }

        @keyframes underlinePulse {
            0%, 100% { width: 46%; opacity: 0.72; }
            50% { width: 72%; opacity: 1; }
        }

        @keyframes cardAuraRotate {
            0% { transform: rotate(0deg); opacity: 0.40; }
            50% { opacity: 0.76; }
            100% { transform: rotate(360deg); opacity: 0.40; }
        }

        @keyframes buttonGradientFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @media (max-width: 900px) {
            .hero-grid, .status-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .risk-hero-row,
            .risk-dashboard-row {
                flex-direction: column;
                align-items: flex-start;
            }

            .risk-percent-display,
            .risk-dashboard-value {
                text-align: left;
                justify-content: flex-start;
            }

            .risk-level-badge {
                min-width: auto;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation: none !important;
                transition: none !important;
            }
        }
        </style>
        """
    )


def landing_page() -> None:
    landing_html = dedent(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                * { box-sizing: border-box; }

                html, body {
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    min-height: 100%;
                    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    color: #FFFFFF;
                    overflow: hidden;
                    background: transparent;
                }

                .stage {
                    position: relative;
                    min-height: 610px;
                    display: grid;
                    place-items: center;
                    padding: 44px 26px 28px;
                    isolation: isolate;
                    background:
                        radial-gradient(circle at 15% 12%, rgba(252,128,25,0.22), transparent 30%),
                        radial-gradient(circle at 85% 18%, rgba(255,159,69,0.12), transparent 30%),
                        radial-gradient(circle at 50% 100%, rgba(252,128,25,0.08), transparent 36%),
                        linear-gradient(135deg, #0B0B0B 0%, #050505 48%, #111111 100%);
                }

                .stage::before {
                    content: "";
                    position: absolute;
                    inset: 0;
                    background:
                        linear-gradient(rgba(252,128,25,0.045) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(252,128,25,0.045) 1px, transparent 1px);
                    background-size: 58px 58px;
                    mask-image: radial-gradient(circle at center, black 0%, transparent 78%);
                    animation: gridDrift 18s linear infinite;
                    z-index: -5;
                }

                .network {
                    position: absolute;
                    inset: 0;
                    width: 100%;
                    height: 100%;
                    opacity: 0.48;
                    z-index: -4;
                }

                .network-line {
                    stroke: rgba(255,159,69,0.26);
                    stroke-width: 1.2;
                    filter: drop-shadow(0 0 8px rgba(252,128,25,0.26));
                }

                .network-node {
                    fill: rgba(252,128,25,0.62);
                    filter: drop-shadow(0 0 12px rgba(252,128,25,0.58));
                    animation: nodePulse 3.8s ease-in-out infinite;
                }

                .hero-card {
                    position: relative;
                    width: min(1010px, 100%);
                    padding: 38px;
                    border-radius: 38px;
                    background:
                        linear-gradient(135deg, rgba(24,24,24,0.92), rgba(5,5,5,0.80)),
                        radial-gradient(circle at top left, rgba(252,128,25,0.16), transparent 32%),
                        radial-gradient(circle at bottom right, rgba(255,159,69,0.10), transparent 32%);
                    border: 1px solid rgba(252,128,25,0.25);
                    box-shadow:
                        0 38px 130px rgba(0,0,0,0.62),
                        0 0 76px rgba(252,128,25,0.14),
                        inset 0 1px 0 rgba(255,255,255,0.10);
                    backdrop-filter: blur(24px);
                    overflow: hidden;
                }

                .hero-card::after {
                    content: "";
                    position: absolute;
                    inset: 0;
                    background: linear-gradient(105deg, transparent 0%, transparent 40%, rgba(255,255,255,0.06) 50%, transparent 61%, transparent 100%);
                    transform: translateX(-115%);
                    animation: cardSheen 7s ease-in-out infinite;
                    pointer-events: none;
                }

                .brand-row {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 10px;
                    margin-bottom: 16px;
                }

                .brand-chip {
                    display: inline-flex;
                    align-items: center;
                    gap: 9px;
                    padding: 8px 13px;
                    border-radius: 999px;
                    color: #fff7ed;
                    background: rgba(252,128,25,0.13);
                    border: 1px solid rgba(252,128,25,0.34);
                    font-size: 12px;
                    font-weight: 800;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                    box-shadow: 0 0 28px rgba(252,128,25,0.14);
                }

                .title {
                    margin: 0 auto;
                    text-align: center;
                    font-size: clamp(64px, 10vw, 132px);
                    line-height: 0.88;
                    font-weight: 950;
                    letter-spacing: -0.08em;
                    background: linear-gradient(90deg, #FFFFFF 0%, #fff7ed 34%, #FF9F45 68%, #FC8019 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    filter: drop-shadow(0 14px 36px rgba(252,128,25,0.16));
                }

                .tagline {
                    max-width: 720px;
                    margin: 22px auto 22px;
                    color: #D1D5DB;
                    text-align: center;
                    font-size: clamp(17px, 2.2vw, 23px);
                    line-height: 1.48;
                    font-weight: 650;
                }

                .hero-grid {
                    display: grid;
                    grid-template-columns: 1.05fr 0.95fr;
                    gap: 18px;
                    align-items: stretch;
                    margin-top: 24px;
                }

                .ecg-panel {
                    position: relative;
                    min-height: 182px;
                    border-radius: 28px;
                    background:
                        linear-gradient(135deg, rgba(11,11,11,0.82), rgba(24,24,24,0.56)),
                        linear-gradient(rgba(252,128,25,0.055) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(252,128,25,0.055) 1px, transparent 1px);
                    background-size: auto, 38px 38px, 38px 38px;
                    border: 1px solid rgba(252,128,25,0.24);
                    box-shadow:
                        inset 0 1px 0 rgba(255,255,255,0.08),
                        0 20px 58px rgba(0,0,0,0.38);
                    overflow: hidden;
                }

                .ecg-svg {
                    position: absolute;
                    inset: 0;
                    width: 100%;
                    height: 100%;
                    padding: 18px;
                }

                .ecg-track {
                    fill: none;
                    stroke: rgba(209,213,219,0.14);
                    stroke-width: 5;
                    stroke-linecap: round;
                    stroke-linejoin: round;
                }

                .ecg-line {
                    fill: none;
                    stroke: #FF9F45;
                    stroke-width: 6;
                    stroke-linecap: round;
                    stroke-linejoin: round;
                    filter: drop-shadow(0 0 12px rgba(255,159,69,0.76));
                    stroke-dasharray: 720;
                    stroke-dashoffset: 720;
                    animation: ecgTrace 3.2s linear infinite;
                }

                .ecg-scan {
                    position: absolute;
                    top: 0;
                    bottom: 0;
                    width: 90px;
                    left: -110px;
                    background: linear-gradient(90deg, transparent, rgba(255,159,69,0.22), transparent);
                    animation: ecgScan 3.2s linear infinite;
                }

                .ecg-label {
                    position: absolute;
                    left: 22px;
                    bottom: 18px;
                    color: #FFFFFF;
                    font-size: 11.5px;
                    font-weight: 850;
                    letter-spacing: 0.035em;
                    background: rgba(11,11,11,0.72);
                    border: 1px solid rgba(252,128,25,0.28);
                    padding: 7px 10px;
                    border-radius: 999px;
                    backdrop-filter: blur(10px);
                }

                .highlights {
                    display: grid;
                    gap: 12px;
                }

                .mini-card {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 17px 18px;
                    min-height: 52px;
                    border-radius: 22px;
                    background: rgba(24,24,24,0.86);
                    border: 1px solid rgba(255,255,255,0.10);
                    color: #FFFFFF;
                    box-shadow:
                        inset 0 1px 0 rgba(255,255,255,0.06),
                        0 16px 44px rgba(0,0,0,0.30);
                    transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
                }

                .mini-card:hover {
                    transform: translateY(-4px);
                    border-color: rgba(255,159,69,0.45);
                    box-shadow:
                        0 24px 64px rgba(0,0,0,0.45),
                        0 0 30px rgba(252,128,25,0.14);
                }

                .mini-icon {
                    width: 34px;
                    height: 34px;
                    border-radius: 12px;
                    display: grid;
                    place-items: center;
                    color: #050505;
                    background: linear-gradient(135deg, #FF9F45, #FC8019);
                    font-size: 16px;
                    font-weight: 950;
                    box-shadow: 0 0 26px rgba(252,128,25,0.20);
                }

                .mini-text {
                    font-size: 15px;
                    font-weight: 850;
                    color: #FFFFFF;
                }

                @keyframes gridDrift {
                    0% { background-position: 0 0, 0 0; }
                    100% { background-position: 58px 58px, 58px 58px; }
                }

                @keyframes nodePulse {
                    0%, 100% { opacity: 0.42; transform: scale(1); }
                    50% { opacity: 1; transform: scale(1.18); }
                }

                @keyframes cardSheen {
                    0%, 62% { transform: translateX(-115%); }
                    100% { transform: translateX(115%); }
                }

                @keyframes ecgTrace {
                    0% { stroke-dashoffset: 720; opacity: 0.55; }
                    15% { opacity: 1; }
                    70% { stroke-dashoffset: 0; opacity: 1; }
                    100% { stroke-dashoffset: -720; opacity: 0.75; }
                }

                @keyframes ecgScan {
                    0% { left: -110px; opacity: 0; }
                    12% { opacity: 1; }
                    88% { opacity: 1; }
                    100% { left: 100%; opacity: 0; }
                }

                @media (max-width: 900px) {
                    .stage {
                        min-height: auto;
                        padding: 26px 16px;
                    }

                    .hero-card {
                        padding: 28px 20px;
                        border-radius: 30px;
                    }

                    .title {
                        font-size: 62px;
                    }

                    .hero-grid {
                        grid-template-columns: 1fr;
                    }

                    .ecg-panel {
                        min-height: 155px;
                    }
                }
            </style>
        </head>
        <body>
            <main class="stage">
                <svg class="network" viewBox="0 0 1200 620" preserveAspectRatio="none">
                    <line class="network-line" x1="160" y1="110" x2="370" y2="220"/>
                    <line class="network-line" x1="370" y1="220" x2="590" y2="120"/>
                    <line class="network-line" x1="590" y1="120" x2="850" y2="210"/>
                    <line class="network-line" x1="850" y1="210" x2="1035" y2="110"/>
                    <line class="network-line" x1="220" y1="480" x2="470" y2="390"/>
                    <line class="network-line" x1="470" y1="390" x2="720" y2="500"/>
                    <line class="network-line" x1="720" y1="500" x2="990" y2="400"/>
                    <circle class="network-node" cx="160" cy="110" r="5"/>
                    <circle class="network-node" cx="370" cy="220" r="5"/>
                    <circle class="network-node" cx="590" cy="120" r="5"/>
                    <circle class="network-node" cx="850" cy="210" r="5"/>
                    <circle class="network-node" cx="1035" cy="110" r="5"/>
                    <circle class="network-node" cx="220" cy="480" r="5"/>
                    <circle class="network-node" cx="470" cy="390" r="5"/>
                    <circle class="network-node" cx="720" cy="500" r="5"/>
                    <circle class="network-node" cx="990" cy="400" r="5"/>
                </svg>

                <section class="hero-card">
                    <div class="brand-row">
                        <div class="brand-chip">AI Healthcare MLOps</div>
                    </div>

                    <h1 class="title">CareGuard AI</h1>

                    <p class="tagline">
                        AI-powered hospital readmission risk prediction for diabetic patients.
                    </p>

                    <div class="hero-grid">
                        <div class="ecg-panel">
                            <svg class="ecg-svg" viewBox="0 0 640 180" preserveAspectRatio="none">
                                <path class="ecg-track" d="M0 105 H70 L100 64 L135 148 L174 26 L220 105 H320 L350 80 L385 128 L425 105 H640"/>
                                <path class="ecg-line" d="M0 105 H70 L100 64 L135 148 L174 26 L220 105 H320 L350 80 L385 128 L425 105 H640"/>
                            </svg>
                            <div class="ecg-scan"></div>
                            <div class="ecg-label">A healthy outside starts from the inside</div>
                        </div>

                        <div class="highlights">
                            <div class="mini-card">
                                <div class="mini-icon">R</div>
                                <div class="mini-text">Recall-Focused ML</div>
                            </div>
                            <div class="mini-card">
                                <div class="mini-icon">M</div>
                                <div class="mini-text">MLOps Pipeline</div>
                            </div>
                            <div class="mini-card">
                                <div class="mini-icon">C</div>
                                <div class="mini-text">Clinical Decision Support</div>
                            </div>
                        </div>
                    </div>
                </section>
            </main>
        </body>
        </html>
        """
    )

    components.html(landing_html, height=640, scrolling=False)

    c1, c2, c3 = st.columns([1, 1.05, 1])
    with c2:
        if st.button("Launch Dashboard", use_container_width=True):
            st.session_state.app_started = True
            st.rerun()


def sidebar() -> None:
    with st.sidebar:
        st.markdown("## CareGuard AI")
        st.caption("Hospital Readmission Risk Scorer")

        st.divider()

        st.markdown("### System Status")
        st.success("Model loaded")
        st.info("FastAPI ready")
        st.warning("Clinical demo only")

        st.divider()

        st.markdown("### MLOps Pipeline")
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

        if st.button("Back to Launch Page"):
            st.session_state.app_started = False
            st.rerun()

        st.caption("Built for hackathon demonstration.")


def hero() -> None:
    render_html(
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
                    <div class="hero-chip">Clinical Risk Scoring</div>
                    <div class="hero-chip">Machine Learning Pipeline</div>
                    <div class="hero-chip">MLflow Tracking</div>
                    <div class="hero-chip">FastAPI + Streamlit</div>
                </div>
            </div>
        </div>
        """
    )


def metrics_panel(metrics: dict) -> None:
    render_html('<div class="section-title">Model Performance Snapshot</div>')

    if not metrics:
        st.info("Run `python src/train.py` to generate model metrics.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    c2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    c3.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
    c4.metric("False Negatives", metrics.get("false_negatives", "N/A"))

    render_html(
        """
        <div class="explanation">
            <b>Why recall matters:</b> In hospital readmission prediction, a false negative means a truly high-risk patient may not receive timely follow-up care.
            This project intentionally prioritizes recall over raw accuracy.
        </div>
        """
    )

    with st.expander("View full metrics JSON"):
        st.json(metrics)


def evaluation_graphs() -> None:
    with st.expander("View Model Evaluation Graphs", expanded=False):
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


def main() -> None:
    st.set_page_config(
        page_title="CareGuard AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    if "app_started" not in st.session_state:
        st.session_state.app_started = False

    if not st.session_state.app_started:
        landing_page()
        return

    sidebar()
    hero()

    sample = load_sample_patient()
    metrics = load_metrics()

    render_html(
        f"""
        <div class="warning-box">
            <b>Clinical Disclaimer:</b> {escape(str(CLINICAL_DISCLAIMER))}
        </div>
        """
    )

    st.write("")

    left, right = st.columns([1.12, 0.88], gap="large")

    with left:
        render_html('<div class="section-title">Patient Intelligence Input</div>')
        render_html(
            """
            <div class="tiny-text">
                Enter patient admission, diagnosis, and diabetes-care information.
                The model will estimate readmission probability and generate a care recommendation.
            </div>
            """
        )

        with st.form("patient_form"):
            tab1, tab2, tab3, tab4 = st.tabs(["Patient", "Admission", "Clinical", "Diabetes Care"])

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

                race = st.selectbox("Race", race_options, index=get_select_index(race_options, sample.get("race", "Asian"), 2))
                gender = st.selectbox("Gender", gender_options, index=get_select_index(gender_options, sample.get("gender", "Male"), 1))
                age = st.selectbox("Age", age_options, index=get_select_index(age_options, sample.get("age", "[40-50)"), 4))

            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    admission_type_id = st.number_input("Admission Type ID", min_value=1, max_value=8, value=int(sample.get("admission_type_id", 1)))
                    admission_source_id = st.number_input("Admission Source ID", min_value=1, max_value=25, value=int(sample.get("admission_source_id", 7)))
                with c2:
                    discharge_disposition_id = st.number_input("Discharge Disposition ID", min_value=1, max_value=30, value=int(sample.get("discharge_disposition_id", 1)))
                    time_in_hospital = st.number_input("Time in hospital", min_value=1, max_value=30, value=int(sample.get("time_in_hospital", 2)))

            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    num_lab_procedures = st.number_input("Number of lab procedures", min_value=0, max_value=150, value=int(sample.get("num_lab_procedures", 5)))
                    num_procedures = st.number_input("Number of procedures", min_value=0, max_value=20, value=int(sample.get("num_procedures", 1)))
                    num_medications = st.number_input("Number of medications", min_value=0, max_value=100, value=int(sample.get("num_medications", 18)))
                    number_diagnoses = st.number_input("Number of diagnoses", min_value=1, max_value=30, value=int(sample.get("number_diagnoses", 8)))
                with c2:
                    number_outpatient = st.number_input("Previous outpatient visits", min_value=0, max_value=50, value=int(sample.get("number_outpatient", 1)))
                    number_emergency = st.number_input("Previous emergency visits", min_value=0, max_value=50, value=int(sample.get("number_emergency", 0)))
                    number_inpatient = st.number_input("Previous inpatient visits", min_value=0, max_value=50, value=int(sample.get("number_inpatient", 2)))

            with tab4:
                c1, c2 = st.columns(2)

                with c1:
                    diag_1 = st.text_input("Diagnosis 1", value=str(sample.get("diag_1", "250.83")))
                    diag_2 = st.text_input("Diagnosis 2", value=str(sample.get("diag_2", "401")))
                    diag_3 = st.text_input("Diagnosis 3", value=str(sample.get("diag_3", "428")))

                    a1c_options = ["None", "Norm", ">7", ">8"]
                    max_glu_options = ["None", "Norm", ">200", ">300"]

                    a1c = st.selectbox("A1C result", a1c_options, index=get_select_index(a1c_options, sample.get("A1Cresult", ">8"), 3))
                    max_glu_serum = st.selectbox("Max glucose serum", max_glu_options, index=get_select_index(max_glu_options, sample.get("max_glu_serum", ">200"), 2))

                with c2:
                    medicine_options = ["No", "Steady", "Up", "Down"]
                    change_options = ["No", "Ch"]
                    diabetes_med_options = ["No", "Yes"]

                    metformin = st.selectbox("Metformin", medicine_options, index=get_select_index(medicine_options, sample.get("metformin", "Steady"), 1))
                    insulin = st.selectbox("Insulin", medicine_options, index=get_select_index(medicine_options, sample.get("insulin", "Up"), 2))
                    change = st.selectbox("Medication change", change_options, index=get_select_index(change_options, sample.get("change", "Ch"), 1))
                    diabetes_med = st.selectbox("Diabetes medication", diabetes_med_options, index=get_select_index(diabetes_med_options, sample.get("diabetesMed", "Yes"), 1))

            submitted = st.form_submit_button("Generate Readmission Risk Score")

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

    with right:
        render_html('<div class="section-title">AI Risk Command Center</div>')

        render_html(
            f"""
            <div class="patient-card">
                <b>Patient:</b> {escape(patient_name)}<br>
                <b>Patient ID:</b> {escape(patient_id)}<br>
                <b>Care Coordinator:</b> {escape(care_coordinator)}
            </div>
            """
        )

        if submitted:
            try:
                with st.spinner("CareGuard AI is analyzing patient risk..."):
                    result = predict_readmission(patient_payload)

                probability = float(result["risk_probability"])
                probability_percent = probability * 100
                dashboard_risk_level, dashboard_risk_color = get_dashboard_risk_level(probability_percent)

                risk_placeholder = st.empty()
                animate_risk_score(
                    placeholder=risk_placeholder,
                    target_percent=probability_percent,
                    color=dashboard_risk_color,
                    risk_level=dashboard_risk_level,
                    prediction=str(result["prediction"]),
                )

                st.progress(min(max(probability, 0.0), 1.0))

                render_html(
                    f"""
                    <div class="status-row">
                        <div class="status-card">
                            <div class="status-value">{float(result["threshold_used"]):.2f}</div>
                            <div class="status-label">Threshold</div>
                        </div>
                        <div class="status-card">
                            <div class="status-value">{float(metrics.get("recall", 0)):.2f}</div>
                            <div class="status-label">Recall</div>
                        </div>
                        <div class="status-card">
                            <div class="status-value">{float(metrics.get("precision", 0)):.2f}</div>
                            <div class="status-label">Precision</div>
                        </div>
                        <div class="status-card">
                            <div class="status-value">{metrics.get("false_negatives", 0)}</div>
                            <div class="status-label">False Negatives</div>
                        </div>
                    </div>
                    """
                )

                render_html(
                    f"""
                    <div class="explanation">
                        <b>Explanation</b><br>
                        {escape(str(result["explanation"]))}
                    </div>
                    """
                )

                render_html(
                    f"""
                    <div class="recommendation">
                        <b>Recommended Care Action</b><br>
                        {escape(str(result["recommendation"]))}
                    </div>
                    """
                )

                render_html(
                    f"""
                    <div class="warning-box">
                        <b>Disclaimer:</b> {escape(str(result["clinical_disclaimer"]))}
                    </div>
                    """
                )

            except FileNotFoundError:
                st.error("Model not found. First run `python src/train.py` from the project root.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        else:
            render_html(
                risk_dashboard_card(
                    percent_value="--",
                    color="#FF9F45",
                    show_details=False,
                )
            )

            render_html(
                """
                <div class="explanation">
                    <b>Demo Tip:</b><br>
                    Use prior inpatient visits, medication change, high A1C, and insulin adjustment to demonstrate a strong readmission-risk scenario.
                </div>
                """
            )

        with st.expander("View Patient JSON Sent to Model"):
            st.json(patient_payload)

    st.write("")

    bottom_left, bottom_right = st.columns([1, 1], gap="large")

    with bottom_left:
        metrics_panel(metrics)

    with bottom_right:
        render_html('<div class="section-title">Technical Transparency</div>')
        render_html(
            """
            <div class="tiny-text">
                CareGuard AI compares models, logs experiments using MLflow, saves model artifacts,
                tunes thresholds for recall, and exposes predictions through both FastAPI and Streamlit.
            </div>
            """
        )
        evaluation_graphs()

    render_html(
        """
        <div class="footer">
            CareGuard AI • Recall-Focused Hospital Readmission Risk Scorer • Hackathon MLOps Demo
        </div>
        """
    )


if __name__ == "__main__":
    main()