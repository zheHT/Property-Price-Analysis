"""
============================================================
  Data-Driven Property Price Analysis & Classification System
  Malaysia Context | Streamlit ML Dashboard
  Author: Product Engineer
============================================================
"""

from pathlib import Path
import pickle
import sys

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PropSense Malaysia",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Minimal premium app-style interface
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  :root {
    --ink: #15171C;
    --muted: #717782;
    --soft: #A7ADB7;
    --line: rgba(21, 23, 28, 0.08);
    --surface: rgba(255, 255, 255, 0.82);
    --surface-strong: #FFFFFF;
    --canvas: #F7F5F1;
    --navy: #172033;
    --blue: #315EAE;
    --green: #2F7D62;
    --amber: #B97830;
    --radius: 22px;
    --shadow: 0 18px 50px rgba(30, 35, 45, 0.08);
  }

  html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
    color: var(--ink);
    letter-spacing: 0;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .stApp {
    background:
      radial-gradient(circle at 16% 0%, rgba(255, 255, 255, 0.95), rgba(255,255,255,0) 32%),
      linear-gradient(180deg, #FAF8F4 0%, #F2F4F6 100%);
  }
  .block-container {
    max-width: 1240px;
    padding: 0.75rem 2.5rem 3.8rem;
  }

  [data-testid="stSidebar"] {
    background: rgba(250, 248, 244, 0.96);
    border-right: 1px solid var(--line);
    box-shadow: 14px 0 40px rgba(31, 35, 42, 0.04);
  }
  [data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.45rem;
  }
  [data-testid="stSidebar"] * {
    color: var(--ink) !important;
  }
  [data-testid="stSidebar"] button { visibility: hidden !important; width: 0 !important; height: 0 !important; }
  [data-testid="stSidebar"] > div > div button { visibility: visible !important; width: auto !important; height: auto !important; }
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
  }
  [data-testid="stSidebar"] h5 {
    font-size: 0.78rem !important;
    color: var(--ink) !important;
    font-weight: 750 !important;
    margin-top: 1.5rem !important;
  }
  [data-testid="stSidebar"] .stSelectbox > div > div,
  [data-testid="stSidebar"] .stNumberInput input {
    min-height: 46px;
    background: rgba(255,255,255,0.72) !important;
    border: 1px solid rgba(21,23,28,0.08) !important;
    border-radius: 16px !important;
    box-shadow: none !important;
  }
  [data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: var(--navy) !important;
  }
  [data-testid="stSidebar"] [data-testid="stTickBar"] {
    color: var(--soft) !important;
  }

  .sidebar-logo {
    font-size: 1.28rem;
    font-weight: 800;
    letter-spacing: 0;
    margin-bottom: 0.25rem;
  }
  .sidebar-tagline {
    font-size: 0.76rem;
    color: var(--muted) !important;
    margin-bottom: 1.5rem;
  }
  .sidebar-divider {
    border: none;
    border-top: 1px solid var(--line);
    margin: 1.45rem 0;
  }

  .hero-block {
    background: rgba(255, 255, 255, 0.68);
    border-radius: 30px;
    padding: 2.25rem 2.85rem;
    margin: 0 0 1.35rem;
    box-shadow: var(--shadow);
    border: 1px solid rgba(255,255,255,0.74);
    backdrop-filter: blur(18px);
  }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    color: var(--muted);
    background: rgba(23, 32, 51, 0.06);
    border-radius: 999px;
    padding: 0.42rem 0.82rem;
    font-size: 0.78rem;
    font-weight: 650;
    margin-bottom: 1.1rem;
  }
  .hero-title {
    max-width: 760px;
    font-size: clamp(2.35rem, 4vw, 4.6rem);
    line-height: 0.96;
    font-weight: 800;
    color: var(--ink);
    margin: 0;
  }
  .hero-subtitle {
    max-width: 590px;
    margin: 1.05rem 0 0;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.65;
  }

  .section-label {
    margin: 2rem 0 0.9rem;
    font-size: 0.82rem;
    color: var(--muted);
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }

  .metric-tile {
    min-height: 142px;
    background: var(--surface);
    border-radius: var(--radius);
    padding: 1.45rem;
    border: 1px solid rgba(255,255,255,0.65);
    box-shadow: 0 14px 36px rgba(31,35,42,0.055);
  }
  .metric-label {
    font-size: 0.73rem;
    color: var(--muted);
    font-weight: 650;
    margin-bottom: 0.85rem;
  }
  .metric-value {
    font-size: 1.55rem;
    line-height: 1.06;
    color: var(--ink);
    font-weight: 780;
    margin-bottom: 0.48rem;
  }
  .metric-sub {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.45;
  }
  .metric-badge-green,
  .metric-badge-amber {
    display: inline-flex;
    margin-top: 0.85rem;
    border-radius: 999px;
    padding: 0.28rem 0.65rem;
    font-size: 0.72rem;
    font-weight: 700;
  }
  .metric-badge-green { background: rgba(47,125,98,0.1); color: var(--green); }
  .metric-badge-amber { background: rgba(185,120,48,0.12); color: var(--amber); }

  .predict-card {
    min-height: 248px;
    border-radius: 28px;
    padding: 2.25rem;
    background: var(--surface-strong);
    box-shadow: var(--shadow);
    border: 1px solid rgba(255,255,255,0.72);
  }
  .predict-card.budget { background: linear-gradient(145deg, #FFFFFF, #EFF8F3); }
  .predict-card.midrange { background: linear-gradient(145deg, #FFFFFF, #EEF4FB); }
  .predict-card.highend { background: linear-gradient(145deg, #FFFFFF, #FBF4EA); }
  .predict-card.luxury { background: linear-gradient(145deg, #FFFFFF, #F6F1EA); }
  .predict-icon {
    width: 54px;
    height: 54px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 18px;
    background: rgba(23,32,51,0.06);
    font-size: 1.55rem;
    margin-bottom: 1.1rem;
  }
  .predict-category {
    font-size: 2.15rem;
    line-height: 1;
    font-weight: 800;
    margin: 0 0 0.55rem;
  }
  .predict-range {
    color: var(--muted);
    font-size: 0.96rem;
  }

  .conf-card {
    min-height: 248px;
    background: rgba(255,255,255,0.78);
    border-radius: 28px;
    padding: 2rem;
    box-shadow: var(--shadow);
    border: 1px solid rgba(255,255,255,0.72);
  }
  .conf-bar-wrap {
    height: 8px;
    background: rgba(23,32,51,0.08);
    border-radius: 999px;
    margin: 0.95rem 0 1.45rem;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--navy), var(--blue));
  }
  .detail-row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.82rem 0;
    border-top: 1px solid var(--line);
  }
  .detail-row span:first-child {
    color: var(--muted);
    font-size: 0.82rem;
    font-weight: 600;
  }
  .detail-row span:last-child {
    color: var(--ink);
    font-size: 0.88rem;
    font-weight: 700;
    text-align: right;
  }

  .suggestion-box,
  .match-box {
    border: none;
    border-radius: 20px;
    padding: 1.25rem 1.45rem;
    margin-top: 1rem;
    line-height: 1.65;
    box-shadow: 0 12px 32px rgba(31,35,42,0.05);
  }
  .suggestion-box { background: #FFF7E8; color: #6E4A1F; }
  .match-box { background: #EEF8F3; color: #245B47; }

  .chart-card {
    background: rgba(255,255,255,0.76);
    border-radius: 28px;
    padding: 1.35rem 1.55rem 0.7rem;
    border: 1px solid rgba(255,255,255,0.72);
    box-shadow: var(--shadow);
  }

  .stButton > button {
    width: 100%;
    min-height: 50px;
    background: var(--navy);
    color: #FFFFFF;
    border: none;
    border-radius: 16px;
    font-weight: 750;
    box-shadow: 0 12px 28px rgba(23,32,51,0.18);
    transition: transform 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease;
  }
  .stButton > button:hover {
    background: #202A3F;
    transform: translateY(-1px);
    box-shadow: 0 16px 34px rgba(23,32,51,0.2);
  }
  [data-testid="stSidebar"] .stButton > button,
  [data-testid="stSidebar"] .stButton > button:focus,
  [data-testid="stSidebar"] .stButton > button:active,
  [data-testid="stSidebar"] .stButton > button:hover {
    width: 100% !important;
    min-height: 50px !important;
    background: var(--navy) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--navy) !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 28px rgba(23,32,51,0.18) !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    display: inline-flex;
    gap: 0.35rem;
    padding: 0.32rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(255,255,255,0.7);
    box-shadow: 0 10px 28px rgba(31,35,42,0.045);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 0.45rem 1rem;
    background: transparent;
    color: var(--muted);
    font-size: 0.86rem;
    font-weight: 700;
  }
  .stTabs [aria-selected="true"] {
    background: var(--navy) !important;
    color: #FFFFFF !important;
  }

  hr { border-color: var(--line) !important; }
  [data-testid="stHorizontalBlock"] { gap: 1rem; }
  @media (max-width: 760px) {
    .block-container { padding: 0.7rem 1rem 2.5rem; }
    .hero-block { padding: 1.7rem 1.45rem; border-radius: 24px; }
    .metric-tile, .predict-card, .conf-card, .chart-card { border-radius: 22px; }
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL + MARKET DATA
# ─────────────────────────────────────────────────────────────

MUKIMS = [
    "Cheras", "Ulu Kelang", "Ampang", "Setapak",
    "Petaling", "Kuala Lumpur Town Centre", "Kuala Lumpur", "Batu"
]

PROPERTY_TYPES = [
    "CONDOMINIUM/APARTMENT",
    "FLAT",
    "LOW-COST FLAT",
    "TOWN HOUSE",
    "LOW-COST HOUSE",
    "CLUSTER HOUSE",
    "DETACHED",
    "1 - 1 1/2 STOREY SEMI-DETACHED",
    "1 - 1 1/2 STOREY TERRACED",
    "2 - 2 1/2 STOREY SEMI-DETACHED",
    "2 - 2 1/2 STOREY TERRACED",
]
TENURE_OPTIONS = ["FREEHOLD", "LEASEHOLD"]

# Mukim average price index (RM '000) for bar chart
MUKIM_AVG_PRICE = {
    "Cheras": 420, "Ulu Kelang": 520, "Ampang": 680,
    "Setapak": 400, "Petaling": 750, "Kuala Lumpur Town Centre": 920,
    "Kuala Lumpur": 890, "Batu": 290
}

MODEL_PATH = APP_DIR / "kl_property_model_rf.pkl"
PREPROCESSOR_PATH = APP_DIR / "property_preprocessor.pkl"

STRATA_PROPERTY_TYPES = {
    "CONDOMINIUM/APARTMENT",
    "FLAT",
    "LOW-COST FLAT",
    "TOWN HOUSE",
}

PROPERTY_CATEGORY_BY_TYPE = {
    property_type: "STRATA" if property_type in STRATA_PROPERTY_TYPES else "LANDED"
    for property_type in PROPERTY_TYPES
}

MUKIM_TO_MODEL_VALUE = {
    "Cheras": "MUKIM CHERAS",
    "Ulu Kelang": "MUKIM ULU KELANG",
    "Ampang": "MUKIM AMPANG",
    "Setapak": "MUKIM SETAPAK",
    "Petaling": "MUKIM PETALING",
    "Kuala Lumpur Town Centre": "KUALA LUMPUR TOWN CENTRE",
    "Kuala Lumpur": "MUKIM KUALA LUMPUR",
    "Batu": "MUKIM BATU",
}

PRICE_CATEGORY_DETAILS = {
    0: ("Budget", "🏠", "budget", "< RM 500,000"),
    1: ("Mid-Range", "🏡", "midrange", "RM 500K – RM 1.2M"),
    2: ("Luxury", "🏛️", "luxury", "> RM 1.2M"),
}

PROPERTY_TYPE_PRICE_MULTIPLIER = {
    "CONDOMINIUM/APARTMENT": 1.00,
    "FLAT": 0.72,
    "LOW-COST FLAT": 0.58,
    "TOWN HOUSE": 1.04,
    "LOW-COST HOUSE": 0.68,
    "CLUSTER HOUSE": 1.18,
    "DETACHED": 1.55,
    "1 - 1 1/2 STOREY SEMI-DETACHED": 1.28,
    "1 - 1 1/2 STOREY TERRACED": 1.08,
    "2 - 2 1/2 STOREY SEMI-DETACHED": 1.38,
    "2 - 2 1/2 STOREY TERRACED": 1.16,
}


def estimate_property_price_k(area_sqft: int, prop_type: str, mukim: str) -> int:
    """Return the budget-analysis estimate in RM '000 for the current property inputs."""
    mukim_average_k = MUKIM_AVG_PRICE.get(mukim, np.median(list(MUKIM_AVG_PRICE.values())))
    area_factor = max(area_sqft, 1) / 1200
    type_factor = PROPERTY_TYPE_PRICE_MULTIPLIER.get(prop_type, 1.0)
    return int(round(mukim_average_k * area_factor * type_factor))

def generate_price_trend(mukim: str, years: int = 5) -> pd.DataFrame:
    """Generate dummy price trend data over time for a given mukim."""
    base_price = MUKIM_AVG_PRICE.get(mukim, 450)
    dates = pd.date_range(end=datetime.today(), periods=years * 12, freq="MS")
    growth = np.linspace(0, 0.25, len(dates))
    noise = np.random.normal(0, 0.02, len(dates))
    prices = base_price * (1 + growth + noise)
    return pd.DataFrame({"Date": dates, "Avg Price (RM '000)": prices.round(1)})

@st.cache_resource(show_spinner=False)
def load_model_assets():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with PREPROCESSOR_PATH.open("rb") as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    return model, preprocessor


def build_feature_matrix(
    area_sqft: int,
    unit_level: int,
    transaction_year: int,
    prop_type: str,
    tenure: str,
    mukim: str,
    model,
    preprocessor: dict,
) -> pd.DataFrame:
    scaler = preprocessor["scaler"]
    ohe = preprocessor["ohe"]
    mukim_mapping = preprocessor["mukim_mapping"]
    feature_names = preprocessor.get("feature_names", list(model.feature_names_in_))

    property_category = PROPERTY_CATEGORY_BY_TYPE.get(prop_type, "STRATA")
    mukim_value = MUKIM_TO_MODEL_VALUE[mukim]

    raw_df = pd.DataFrame([{
        "Effective Area (sqft)": area_sqft,
        "Unit Level": unit_level,
        "Property Type": prop_type,
        "Property Category": property_category,
        "Tenure": tenure,
        "Mukim": mukim_value,
        "Transaction Year": transaction_year,
    }])

    raw_df["Mukim_Rank"] = raw_df["Mukim"].map(mukim_mapping)
    raw_df["Mukim_Rank"] = raw_df["Mukim_Rank"].fillna(np.median(list(mukim_mapping.values())))

    num_cols = list(scaler.feature_names_in_)
    scaled_num = pd.DataFrame(
        scaler.transform(raw_df[num_cols]),
        columns=num_cols,
        index=raw_df.index,
    )

    cat_cols = list(ohe.feature_names_in_)
    encoded = pd.DataFrame(
        ohe.transform(raw_df[cat_cols]),
        columns=ohe.get_feature_names_out(cat_cols),
        index=raw_df.index,
    )

    features = pd.concat([raw_df[["Mukim_Rank"]], scaled_num, encoded], axis=1)
    features["Log_Area"] = np.log1p(raw_df["Effective Area (sqft)"])
    features["Year_Rank"] = raw_df["Transaction Year"] * features["Mukim_Rank"]
    features["Area_Prestige"] = features["Log_Area"] * features["Mukim_Rank"]
    return features.reindex(columns=feature_names, fill_value=0)


def classify_property(
    area_sqft: int,
    unit_level: int,
    transaction_year: int,
    prop_type: str,
    tenure: str,
    mukim: str,
) -> dict:
    model, preprocessor = load_model_assets()
    features = build_feature_matrix(
        area_sqft, unit_level, transaction_year, prop_type, tenure, mukim, model, preprocessor
    )

    predicted_class = int(model.predict(features)[0])
    category, icon, color_class, price_range = PRICE_CATEGORY_DETAILS[predicted_class]
    estimated_rm_k = estimate_property_price_k(area_sqft, prop_type, mukim)

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

    return {
        "category": category,
        "icon": icon,
        "color_class": color_class,
        "price_range": price_range,
        "estimated_rm_k": estimated_rm_k,
        "confidence": round(confidence, 2),
    }

def get_affordable_mukims(budget_k: int, area_sqft: int, prop_type: str) -> list[str]:
    """Return mukims whose recalculated property estimate is within budget."""
    return [
        mukim_name
        for mukim_name in MUKIMS
        if estimate_property_price_k(area_sqft, prop_type, mukim_name) <= budget_k
    ]


# ─────────────────────────────────────────────────────────────
# SIDEBAR — User Inputs
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand identity
    st.markdown('<div class="sidebar-logo">PropSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Malaysia Property Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("##### Property Details")

    # Property Area Slider
    area_sqft = st.slider(
        "Property Area (sqft)",
        min_value=400,
        max_value=5000,
        value=1200,
        step=50,
        help="Total built-up area of the property"
    )

    unit_level = st.number_input(
        "Unit Level",
        min_value=0,
        max_value=150,
        value=0,
        step=1,
        help="Use 0 for landed property or ground floor"
    )

    current_year = datetime.today().year
    if "transaction_year" not in st.session_state:
        st.session_state.transaction_year = current_year

    transaction_year = st.number_input(
        "Transaction Year",
        min_value=2000,
        max_value=current_year,
        step=1,
        key="transaction_year",
    )

    # Property Type Dropdown
    prop_type = st.selectbox(
        "Property Type",
        options=PROPERTY_TYPES,
        index=0,
    )

    # Tenure Dropdown
    tenure = st.selectbox(
        "Tenure",
        options=TENURE_OPTIONS,
        index=0,
    )

    # Mukim Dropdown
    mukim = st.selectbox(
        "Mukim / Area",
        options=MUKIMS,
        index=4,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Optional Budget Slider
    st.markdown("##### Budget")
    enable_budget = st.checkbox("Set my budget limit", value=False)
    budget_k = None
    if enable_budget:
        budget_k = st.slider(
            "Max Budget (RM)",
            min_value=100_000,
            max_value=2_000_000,
            value=500_000,
            step=50_000,
            format="RM %d",
        )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Predict Button
    predict_clicked = st.button("Analyse Property")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.68rem;color:rgba(255,255,255,0.28);text-align:center;line-height:1.6;">'
        'Powered by trained Random Forest model.<br>Preprocessing bundle loaded locally.</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────

# ── Emergency reopen button (always visible) ──
components.html("""
<div style="position: fixed; top: 8px; left: 18px; z-index: 99999;">
  <button id="open-sidebar-btn" style="
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    padding: 0;
    background: rgba(255, 255, 255, 0.78);
    color: #172033;
    border: 1px solid rgba(23, 32, 51, 0.08);
    border-radius: 14px;
    cursor: pointer;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0 12px 30px rgba(31, 35, 42, 0.08);
    backdrop-filter: blur(14px);
    transition: transform 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease;
  ">
    <span aria-hidden="true" style="line-height:1;">☰</span>
    <span style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;">Open sidebar</span>
  </button>
</div>
<script>
  document.getElementById('open-sidebar-btn').addEventListener('click', () => {
    const storage = window.parent.localStorage;
    const keys = [];
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (key && (key.toLowerCase().includes('sidebar') || key.toLowerCase().startsWith('streamlit:'))) {
        keys.push(key);
      }
    }
    keys.forEach(key => storage.removeItem(key));
    window.parent.location.reload();
  });
</script>
""", height=58)

# ── Hero Header ──
st.markdown("""
<div class="hero-block">
  <div class="hero-badge">Malaysia Property Intelligence</div>
  <div class="hero-title">Know the market tier before you move.</div>
  <p class="hero-subtitle">A focused decision screen for classifying Kuala Lumpur residential properties using your trained Random Forest model.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TOP STAT TILES — Quick market snapshot
# ─────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.columns(4)

with t1:
    st.markdown("""
    <div class="metric-tile">
      <div class="metric-label">Average price</div>
      <div class="metric-value">RM 512K</div>
      <div class="metric-sub">Across all mukims</div>
      <span class="metric-badge-green">↑ +4.2% YoY</span>
    </div>""", unsafe_allow_html=True)

with t2:
    st.markdown("""
    <div class="metric-tile">
      <div class="metric-label">Listings tracked</div>
      <div class="metric-value">18,340</div>
      <div class="metric-sub">Active this month</div>
      <span class="metric-badge-green">↑ +12% MoM</span>
    </div>""", unsafe_allow_html=True)

with t3:
    st.markdown("""
    <div class="metric-tile">
      <div class="metric-label">Top mukim</div>
      <div class="metric-value">KL Centre</div>
      <div class="metric-sub">Highest avg. price</div>
      <span class="metric-badge-amber">RM 920K avg.</span>
    </div>""", unsafe_allow_html=True)

with t4:
    st.markdown("""
    <div class="metric-tile">
      <div class="metric-label">Budget hotspot</div>
      <div class="metric-value">Batu / Cheras</div>
      <div class="metric-sub">Most affordable areas</div>
      <span class="metric-badge-green">< RM 430K avg.</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PREDICTION SECTION + CONFIDENCE
# ─────────────────────────────────────────────────────────────
st.markdown('<div id="prediction-result" class="section-label">Prediction Result</div>', unsafe_allow_html=True)

result = None

if predict_clicked:
    with st.spinner("Analyzing property..."):
        try:
            result = classify_property(area_sqft, unit_level, transaction_year, prop_type, tenure, mukim)
        except Exception as exc:
            st.error(f"Model prediction failed: {exc}")

if result:
    st.markdown('<div id="prediction-card-start"></div>', unsafe_allow_html=True)
    components.html("""
    <script>
      const scrollToResult = () => {
        const doc = window.parent.document;
        const target = doc.getElementById("prediction-card-start");
        if (!target) return;

        const scrollables = [
          doc.scrollingElement,
          doc.documentElement,
          doc.body,
          ...Array.from(doc.querySelectorAll("section, main, div"))
            .filter((el) => el.scrollHeight > el.clientHeight + 40)
        ].filter(Boolean);

        scrollables.forEach((el) => {
          const isDocument = el === doc.scrollingElement || el === doc.documentElement || el === doc.body;
          const containerTop = isDocument ? 0 : el.getBoundingClientRect().top;
          const currentTop = isDocument ? window.parent.scrollY : el.scrollTop;
          const targetTop = target.getBoundingClientRect().top;
          const top = currentTop + targetTop - containerTop - 18;

          try {
            el.scrollTo({ top, behavior: "smooth" });
          } catch {
            el.scrollTop = top;
          }
        });
      };

      window.setTimeout(scrollToResult, 300);
      window.setTimeout(scrollToResult, 900);
      window.setTimeout(scrollToResult, 1400);
    </script>
    """, height=1)

    pred_col, conf_col = st.columns([1.1, 1], gap="large")

    with pred_col:
        st.markdown(f"""
        <div class="predict-card {result['color_class']}">
          <div class="predict-icon">{result['icon']}</div>
          <div class="predict-category">{result['category']}</div>
          <div class="predict-range">{result['price_range']}</div>
          <br>
          <div style="font-size:0.88rem;color:#717782;margin-top:0.35rem;">
            Estimated: <strong>RM {int(result['estimated_rm_k']):,}K</strong> &nbsp;·&nbsp; {tenure} &nbsp;·&nbsp; {mukim}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with conf_col:
        conf_pct = int(result['confidence'] * 100)

        st.markdown(f"""
        <div class="conf-card">
          <div class="metric-label">Model confidence</div>
          <div class="metric-value">{conf_pct}%</div>
          <div class="conf-bar-wrap">
            <div class="conf-bar-fill" style="width:{conf_pct}%;"></div>
          </div>
          <div class="detail-row">
            <span>Property type</span>
            <span>{prop_type}</span>
          </div>
          <div class="detail-row">
            <span>Built-up area</span>
            <span>{area_sqft:,} sqft</span>
          </div>
          <div class="detail-row">
            <span>Mukim</span>
            <span>{mukim}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Budget Mismatch Suggestion ──
if result and enable_budget and budget_k is not None:
    budget_k_display = budget_k / 1000
    estimated_price = result["estimated_rm_k"]

    if estimated_price > budget_k_display:
        affordable = get_affordable_mukims(int(budget_k_display), area_sqft, prop_type)
        affordable_str = ", ".join(affordable) if affordable else "None found"
        st.markdown(f"""
        <div class="suggestion-box">
          <strong>Budget Mismatch Detected</strong><br>
          Your selected property in <strong>{mukim}</strong> is estimated at
          <strong>RM {int(estimated_price):,}K</strong>, which exceeds your budget of
          <strong>RM {int(budget_k_display):,}K</strong>.<br><br>
          <strong>Suggested mukims within your budget:</strong> {affordable_str}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="match-box">
          <strong>Within Budget</strong><br>
          Great news — this property's estimated price of <strong>RM {int(estimated_price):,}K</strong>
          fits comfortably within your budget of <strong>RM {int(budget_k_display):,}K</strong>.
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# VISUALIZATIONS — Tabs for line chart & bar chart
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Market Visualizations</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Price Trend", "Mukim Comparison"])

# ── TAB 1: Price Trend Line Chart ──
with tab1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    trend_years = st.select_slider(
        "Select time range",
        options=[1, 2, 3, 5],
        value=3,
        format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}",
    )

    trend_df = generate_price_trend(mukim, years=trend_years)

    fig_line = px.line(
        trend_df,
        x="Date",
        y="Avg Price (RM '000)",
        title=f"Average Property Price Trend — {mukim}",
        labels={"Avg Price (RM '000)": "Avg Price (RM '000)", "Date": ""},
    )
    fig_line.update_traces(
        line=dict(color="#2E4DB5", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(46,77,181,0.07)",
        mode="lines",
    )
    fig_line.update_layout(
        font_family="DM Sans",
        title_font_family="Syne",
        title_font_size=15,
        title_font_color="#1A1D2E",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(gridcolor="rgba(26,29,46,0.06)", showline=False),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: Mukim Comparison Bar Chart ──
with tab2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    mukim_df = pd.DataFrame({
        "Mukim": list(MUKIM_AVG_PRICE.keys()),
        "Avg Price (RM '000)": list(MUKIM_AVG_PRICE.values()),
    }).sort_values("Avg Price (RM '000)", ascending=True)

    # Highlight the selected mukim
    mukim_df["Color"] = mukim_df["Mukim"].apply(
        lambda m: "#2E4DB5" if m == mukim else "#B0BEC5"
    )

    fig_bar = go.Figure(go.Bar(
        x=mukim_df["Avg Price (RM '000)"],
        y=mukim_df["Mukim"],
        orientation="h",
        marker_color=mukim_df["Color"],
        text=mukim_df["Avg Price (RM '000)"].apply(lambda v: f"RM {v}K"),
        textposition="outside",
        textfont=dict(family="DM Sans", size=11, color="#1A1D2E"),
    ))
    fig_bar.update_layout(
        title="Average Property Price by Mukim (Top 10)",
        font_family="DM Sans",
        title_font_family="Syne",
        title_font_size=15,
        title_font_color="#1A1D2E",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(26,29,46,0.06)", showline=False, title=""),
        yaxis=dict(showgrid=False, showline=False, title=""),
        margin=dict(l=0, r=60, t=40, b=0),
        showlegend=False,
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Small annotation
    st.caption(f"Highlighted bar = your selected mukim: **{mukim}**  |  Prices shown are average estimates.")
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.5rem 0 0.5rem;border-top:1px solid rgba(26,29,46,0.08);">
  <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#1B2B6B;">PropSense</span>
  <span style="font-size:0.78rem;color:#8A91B0;margin-left:0.5rem;">· Malaysia Property Intelligence · Demo Build</span><br>
  <span style="font-size:0.72rem;color:#B0B8D0;">Data shown is synthetic and for UI demonstration purposes only. No real property transactions are reflected.</span>
</div>
""", unsafe_allow_html=True)
