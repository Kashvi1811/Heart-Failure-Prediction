# app.py — Dark theme, high-contrast hero, uniform two-column inputs,
# safe beating-heart animation (flash-hardened), colored primary button, bold results with hover,
# Platelets min=10,000 per µL, white inputs & dropdown with visible chevron,
# dark metric text, visible typing caret, and anti-flash layer isolation.

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import joblib # type: ignore
import base64
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer # type: ignore

# ---------------------------
# Setup and model
# ---------------------------

st.set_page_config(page_title="Heart Failure Prediction", layout="wide")
def log_transform_func(x):
    return np.log1p(x)

@st.cache_resource  # ✅ cache model so it's not reloaded every run
def load_model():
    return joblib.load("xgboost_pipeline.pkl")

model = load_model()

IMG_PATH = "image.png"
HERO_BG  = "bg3.jpg"


# ---------------------------
# Utilities
# ---------------------------
def b64_of(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

b64_bg = b64_of(HERO_BG) if Path(HERO_BG).exists() else ""

# ---------------------------
# Styles
# ---------------------------
st.markdown(f"""
<style>
:root{{
  --hero-red:#cb5454; 
  --hero-teal:#0f6868;
  --bg:#dbe4e7;                 /* slightly darker bg to mask any 1-frame gaps */
  --panel:#ffffff;
  --panel-soft:#eef6f6;
  --text:#0b3f3f; 
  --muted:#5f7575;
  --radius-xl:18px; 
  --shadow:0 8px 22px rgba(15,104,104,.14);
}}
            /* ✅ Add this block */
* {{
  transition: none !important;
}}

html, body, [data-testid="stAppViewContainer"]{{ background: var(--bg); }}
.block-container{{ padding-top:0 !important; max-width:1200px; }}

/* ================= Hero (flash-hardened) ================= */
.hero {{
  position: relative;
  background-color:#1a3f3f;           /* solid fallback, no transparency */
  {"background: url('data:image/jpeg;base64," + b64_bg + "');" if b64_bg else "background: linear-gradient(140deg, rgba(203,84,84,.9), rgba(15,104,104,.9));"}
  background-size: cover; background-position: center; color:#fff;
  padding: 72px 36px 64px; border-radius: 0 0 var(--radius-xl) var(--radius-xl);
  margin: 0 -1rem 22px; box-shadow: 0 12px 32px rgba(15,104,104,.22);

  contain: paint;                      /* isolate paints */
  transform: translateZ(0);            /* own compositor layer */
  will-change: transform, opacity;
}}
.hero::before{{
  content:""; position:absolute; inset:0;
  background: linear-gradient(110deg, rgba(0,0,0,.58), rgba(0,0,0,.48));
  border-radius: inherit;
  backface-visibility: hidden;
  transform: translateZ(0);
  will-change: opacity;
}}
.hero::after{{
  content:""; position:absolute; left:0; right:0; top:64px; height:128px;
  background: linear-gradient(180deg, rgba(0,0,0,.55), rgba(0,0,0,.25));
  filter: blur(.5px);
  border-radius:0 0 var(--radius-xl) var(--radius-xl);
  pointer-events:none;

  backface-visibility: hidden;
  transform: translateZ(0);
  will-change: opacity;
}}
.hero .content{{ position:relative; z-index:2; }}
.hero h1{{ margin:0; font-weight:900; font-size:clamp(42px, 6vw, 72px); color:#fff; text-shadow:0 2px 6px rgba(0,0,0,.45); }}
.hero p{{ margin-top:10px; max-width:1100px; color:#F5F7F8; font-weight:700; font-size:clamp(16px, 2.2vw, 22px); line-height:1.55; text-shadow:0 3px 10px rgba(0,0,0,.6), 0 1px 2px rgba(0,0,0,.45); }}

/* ================= Cards & labels ================= */
.card{{ background:var(--panel); border-radius:var(--radius-xl); box-shadow:var(--shadow); padding:20px; border:1px solid rgba(15,104,104,.08);
  backface-visibility: hidden; transform: translateZ(0); will-change: transform, box-shadow; }}
.card.soft{{ background:var(--panel-soft); }}
.section-title{{ color:var(--text); font-weight:900; font-size:26px; margin:0 0 12px; }}
.section-title:after{{ content:""; display:block; width:70px; height:4px; border-radius:3px; background:linear-gradient(90deg,var(--hero-red),var(--hero-teal)); margin-top:8px; }}
.stNumberInput label, .stSelectbox label{{ font-weight:800 !important; color:var(--text) !important; }}

/* ================= Inputs — white + dark text + visible caret ================= */
.stNumberInput input{{
  background:#ffffff !important;
  color:#0b3f3f !important;
  caret-color:#0b3f3f !important;   /* visible cursor */
  cursor:text;                       /* I‑beam inside inputs */
  backface-visibility: hidden; transform: translateZ(0); will-change: transform, box-shadow;
}}
.stNumberInput input::placeholder{{ color:#8aa2a2 !important; opacity:1 !important; }}

/* Select control (closed) */
.stSelectbox div[data-baseweb="select"] > div{{
  background:#ffffff !important; color:#0b3f3f !important;
  border:1px solid #bcdede !important; border-radius:12px !important;
  cursor:pointer;
  backface-visibility: hidden; transform: translateZ(0); will-change: transform, box-shadow;
}}
.stSelectbox div[data-baseweb="select"] span{{ color:#0b3f3f !important; }}

/* Visible chevron for select */
.stSelectbox div[data-baseweb="select"] svg{{ color:#0b3f3f !important; opacity:1 !important; }}
.stSelectbox div[data-baseweb="select"] > div::after{{
  content:""; position:absolute; right:14px; top:50%; width:0; height:0; transform:translateY(-50%);
  border-left:6px solid transparent; border-right:6px solid transparent; border-top:7px solid #0b3f3f; pointer-events:none;
}}

/* Dropdown MENU — pure white with dark text */
[data-baseweb="menu"]{{ 
  background:#ffffff !important; color:#0b3f3f !important;
  border:1px solid #d9e8e8 !important; box-shadow: 0 12px 28px rgba(0,0,0,.12) !important;
}}
[data-baseweb="menu"] li, [data-baseweb="menu"] div[role="option"]{{ background:#ffffff !important; color:#0b3f3f !important; cursor:pointer; }}
[data-baseweb="menu"] li:hover, [data-baseweb="menu"] div[role="option"]:hover{{ background:#f3fbfb !important; color:#0b3f3f !important; }}
[data-baseweb="menu"] li[aria-selected="true"], [data-baseweb="menu"] div[role="option"][aria-selected="true"]{{ background:#e9f6f6 !important; color:#0b3f3f !important; }}

/* Focus rings */
.stNumberInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within{{
  outline:none !important; border-color:#cb5454 !important; box-shadow:0 0 0 3px rgba(203,84,84,.18) !important;
}}

/* ================= Metric — dark text ================= */
[data-testid="stMetric"] [data-testid="stMetricValue"]{{ color:#0b3f3f !important; text-shadow:none !important; }}
[data-testid="stMetric"] [data-testid="stMetricDelta"]{{ color:#0b3f3f !important; }}
[data-testid="stMetricLabel"]{{ color:#0b3f3f !important; opacity:.85 !important; }}

/* ================= Buttons ================= */
.stButton > button[kind="primary"]{{
  width:100%; color:#ffffff; border:none; border-radius:12px; padding:14px 18px; font-weight:900; letter-spacing:.3px;
  background: linear-gradient(135deg, #cb5454 0%, #0f6868 100%) !important;
  box-shadow: 0 12px 26px rgba(15,104,104,.25);
  transition: transform .06s ease, box-shadow .15s ease, filter .15s ease;
}}
.stButton > button[kind="primary"]:hover{{ filter:brightness(1.03); box-shadow:0 16px 34px rgba(15,104,104,.3); }}
.stButton > button[kind="primary"]:active{{ transform: translateY(1px); }}

/* ================= Beating Heart (flash-hardened) ================= */
.heart-wrap{{
  position:relative; display:inline-block; width:100%; max-width:520px;
  overflow:hidden;                      /* clip edges while scaling */
  border-radius:14px;
  background:#142f2f;                   /* solid fallback under image */
  contain: paint;                       /* isolate repaints */
  transform: translateZ(0);             /* GPU layer */
  will-change: transform;
}}

/* Tiny scale to minimize any edge gap; use pulse-shadow fallback if needed */
@keyframes heartBeatSafe {{
  0%   {{ transform: scale(1);    }}
  25%  {{ transform: scale(1.012); }}
  50%  {{ transform: scale(1);    }}
  75%  {{ transform: scale(1.012); }}
  100% {{ transform: scale(1);    }}
}}
@keyframes heartShadow {{
  0%,100% {{ box-shadow: 0 8px 18px rgba(203,84,84,.18); opacity:1; }}
  50%     {{ box-shadow: 0 14px 28px rgba(203,84,84,.28); opacity:.98; }}
}}
.heart-wrap img{{
  width:100%; height:auto; border-radius:14px;
  animation: heartBeatSafe 1.8s ease-in-out infinite;
  transform-origin:50% 50%;
  transform: translateZ(0);
  will-change: transform;
  backface-visibility: hidden;
  background: transparent !important; filter:none !important;
}}
/* To disable scale completely, add class="pulse-shadow" to the img element: */
.heart-wrap img.pulse-shadow{{ animation: heartShadow 1.8s ease-in-out infinite !important; }}

/* ================= Prediction Card ================= */
.prediction-card{{ position:relative; border-radius:16px; padding:22px 24px; text-align:center; font-size:24px; font-weight:900; letter-spacing:.5px; box-shadow:0 10px 28px rgba(15,104,104,.18); transform:translateY(0); transition:transform .18s ease, box-shadow .18s ease, filter .18s ease; overflow:hidden; backface-visibility:hidden; }}
.prediction-card::before{{ content:""; width:0; height:0; display:none; }}
.prediction-card::after{{ content:""; position:absolute; inset:-40%; background:linear-gradient(120deg, rgba(255,255,255,0) 40%, rgba(255,255,255,.22) 50%, rgba(255,255,255,0) 60%); transform:translateX(-120%) rotate(8deg); transition:transform .6s ease; pointer-events:none; }}
.prediction-card:hover{{ transform:translateY(-4px); box-shadow:0 18px 48px rgba(15,104,104,.28); filter:brightness(1.02); }}


</style>
""", unsafe_allow_html=True)

# ---------------------------
# Hero
# ---------------------------
st.markdown("""
<div class="hero">
  <div class="content">
    <h1>CARDIAC HEALTH</h1>
    <p>Advanced predictive analysis for heart failure risk assessment using comprehensive clinical parameters and cutting‑edge algorithms.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Session state defaults
# ---------------------------
for key in ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets",
            "serum_creatinine", "serum_sodium", "time"]:
    if key not in st.session_state:
        st.session_state[key] = None
for key in ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ---------------------------
# Layout: Info + Form
# ---------------------------
left, right = st.columns([1.05, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Heart Failure Prediction</div>', unsafe_allow_html=True)

    if Path(IMG_PATH).exists():
        ext = "png" if IMG_PATH.lower().endswith("png") else "jpeg"
        b64_img = b64_of(IMG_PATH)
        # To eliminate scale entirely, add class="pulse-shadow" below
        st.markdown(f"""
        <div class="heart-wrap">
          <img src="data:image/{ext};base64,{b64_img}" alt="Heart image">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.image(IMG_PATH, use_container_width=True)

   
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Data Input</div>', unsafe_allow_html=True)

    feature_names = [
        "age", "anaemia", "creatinine_phosphokinase", "diabetes",
        "ejection_fraction", "high_blood_pressure", "platelets",
        "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
    ]

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=1, max_value=120,
                              value=st.session_state.age, placeholder="Enter age", key="age_input")
        st.session_state.age = age
    with c2:
        sex_choice = st.selectbox("Sex", ["F", "M", ""], index=2, key="sex_input")
        st.session_state.sex = sex_choice
        sex = 1 if sex_choice == "M" else 0 if sex_choice == "F" else None

    c3, c4 = st.columns(2)
    with c3:
        anaemia_choice = st.selectbox("Anaemia", ["No", "Yes", ""], index=2, key="anaemia_input")
        st.session_state.anaemia = anaemia_choice
        anaemia = 1 if anaemia_choice == "Yes" else 0 if anaemia_choice == "No" else None
    with c4:
        ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=1, max_value=100,
                                            value=st.session_state.ejection_fraction, placeholder="e.g., 40", key="ef_input")
        st.session_state.ejection_fraction = ejection_fraction

    c5, c6 = st.columns(2)
    with c5:
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value = 20.0,
                                           value=st.session_state.serum_creatinine, placeholder="e.g., 1.0", key="scr_input")
        st.session_state.serum_creatinine = serum_creatinine
    with c6:
        serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=200,
                                       value=st.session_state.serum_sodium, placeholder="e.g., 140", key="sodium_input")
        st.session_state.serum_sodium = serum_sodium

    c7, c8 = st.columns(2)
    with c7:
        creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (U/L)", min_value=0,
                                                   value=st.session_state.creatinine_phosphokinase, placeholder="e.g., 100", key="cpk_input")
        st.session_state.creatinine_phosphokinase = creatinine_phosphokinase
    with c8:
        platelets = st.number_input("Platelets (/µL)", min_value=10000,
                                    value=st.session_state.platelets,
                                    placeholder="e.g., 200000", key="platelets_input")
        st.session_state.platelets = platelets

    c9, c10 = st.columns(2)
    with c9:
        smoking_choice = st.selectbox("Smoking", ["No", "Yes", ""], index=2, key="smoking_input")
        st.session_state.smoking = smoking_choice
        smoking = 1 if smoking_choice == "Yes" else 0 if smoking_choice == "No" else None
    with c10:
        high_blood_pressure_choice = st.selectbox("High Blood Pressure", ["No", "Yes", ""], index=2, key="hbp_input")
        st.session_state.high_blood_pressure = high_blood_pressure_choice
        high_blood_pressure = 1 if high_blood_pressure_choice == "Yes" else 0 if high_blood_pressure_choice == "No" else None

    c11, c12 = st.columns(2)
    with c11:
        diabetes_choice = st.selectbox("Diabetes", ["No", "Yes", ""], index=2, key="diabetes_input")
        st.session_state.diabetes = diabetes_choice
        diabetes = 1 if diabetes_choice == "Yes" else 0 if diabetes_choice == "No" else None
    with c12:
        time = st.number_input("Follow-up Period (days)", min_value=0,
                               value=st.session_state.time, placeholder="e.g., 100", key="time_input")
        st.session_state.time = time

    st.divider()
    predict_clicked = st.button("PREDICT HEART FAILURE RISK", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Prediction and results
# ---------------------------
if 'predict_clicked' not in st.session_state:
    st.session_state.predict_clicked = False
if predict_clicked:
    st.session_state.predict_clicked = True

if st.session_state.predict_clicked:
    values = [
        age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
        high_blood_pressure, platelets, serum_creatinine, serum_sodium,
        sex, smoking, time
    ]
    if any(v is None for v in values):
        st.warning("Please fill in all the input fields before predicting.")
    else:
        feature_names = [
            "age", "anaemia", "creatinine_phosphokinase", "diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
        ]
        X = pd.DataFrame([values], columns=feature_names)
        pred = model.predict(X)[0]
        prob = float(model.predict_proba(X)[0][1])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
            if pred == 1:
                st.markdown("""
                    <div class="prediction-card" style="background-color:#f8d7da;color:#721c24;">
                        ⚠️ HIGH RISK OF HEART FAILURE
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="prediction-card" style="background-color:#d1ecf1;color:#0c5460;">
                        ✅ LOW RISK OF HEART FAILURE
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Probability</div>', unsafe_allow_html=True)
            st.metric(label="", value=f"{prob:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
