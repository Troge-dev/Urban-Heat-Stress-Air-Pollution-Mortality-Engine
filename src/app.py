"""
================================================================================
Project CCHAIN: Urban Heat-Health & Air Pollution Interactive Intelligence Hub
Streamlit Dashboard: src/app.py
--------------------------------------------------------------------------------
Interactive decision-support platform for Philippine Local Government Units (LGUs):
1. Real-Time Climate Hazard Early Warning System (EWS) Risk Simulator
2. City-Disaggregated Spatial Vulnerability & UHI Explorer
3. Diagnostic Non-Linear Exposure-Response Curves & SHAP Explainability
================================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page configuration
st.set_page_config(
    page_title="Project CCHAIN | Heat & Pollution Health Engine",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3d59; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 20px; }
    .metric-card { background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 5px solid #17a2b8; }
    .alert-normal { background-color: #d4edda; color: #155724; padding: 12px; border-radius: 6px; border-left: 5px solid #28a745; }
    .alert-caution { background-color: #fff3cd; color: #856404; padding: 12px; border-radius: 6px; border-left: 5px solid #ffc107; }
    .alert-warning { background-color: #ffe5d0; color: #a84200; padding: 12px; border-radius: 6px; border-left: 5px solid #fd7e14; }
    .alert-danger { background-color: #f8d7da; color: #721c24; padding: 12px; border-radius: 6px; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_master_data():
    df = pd.read_csv("data/processed_cchain_master.csv")
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    return df


@st.cache_resource
def load_models():
    models = {
        "Cardiorespiratory Total": joblib.load("output/models/lightgbm_rate_cardiorespiratory_per_100k.joblib"),
        "Ischemic Heart Disease": joblib.load("output/models/lightgbm_rate_ihd_per_100k.joblib"),
        "Hypertensive Heart Disease": joblib.load("output/models/lightgbm_rate_hhd_per_100k.joblib"),
        "Asthma Mortality": joblib.load("output/models/lightgbm_rate_asthma_per_100k.joblib")
    }
    return models


try:
    df_master = load_master_data()
    models = load_models()
    model_cardio = models["Cardiorespiratory Total"]
except Exception as e:
    st.error(f"Error loading master dataset or models: {e}. Please run `python src/data_processing.py` and `python src/train_model.py` first.")
    st.stop()


# Sidebar Navigation
st.sidebar.title("🌍 Project CCHAIN Hub")
st.sidebar.markdown("**Planetary Health & Climate AI in the Philippines**")
app_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "🚨 Heat-Health Early Warning Simulator",
        "🗺️ Municipal Spatial Vulnerability",
        "🔬 Epidemiological GAM & SHAP Insights",
        "📊 Model Benchmarks & Test Reports"
    ]
)

# -----------------------------------------------------------------------------
# 1. Early Warning Simulator
# -----------------------------------------------------------------------------
if app_mode == "🚨 Heat-Health Early Warning Simulator":
    st.markdown('<div class="main-header">🚨 Heat-Health Early Warning System (EWS) Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Forecast municipal weekly cardiorespiratory mortality spikes given simulated temperature and particulate anomalies.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Climate Hazard Inputs")
        
        city_list = df_master["adm3_en"].unique().tolist()
        selected_city = st.selectbox("Select Target City / Municipality", city_list, index=city_list.index("City of Mandaluyong") if "City of Mandaluyong" in city_list else 0)
        
        heat_idx_input = st.slider("Weekly 95th Percentile Heat Index (°C)", min_value=25.0, max_value=50.0, value=38.5, step=0.5)
        pm25_input = st.slider("Weekly Mean PM2.5 (µg/m³)", min_value=5.0, max_value=100.0, value=35.0, step=1.0)
        extreme_days = st.slider("Consecutive Hot Days in Week (HI ≥ 37°C)", min_value=0, max_value=7, value=3)
        season_input = st.selectbox("Climate Season", ["Hot-Dry (Tag-init)", "Wet (Tag-ulan)", "Cool-Dry (Tag-lamig)"])

        # Fetch city static metadata
        city_meta = df_master[df_master["adm3_en"] == selected_city].iloc[-1]
        
    with col2:
        st.subheader("🎯 Simulated Health Outcomes & Alert Matrix")

        # Build feature vector matching model columns
        input_data = city_meta.copy()
        input_data["heat_index_mean"] = heat_idx_input - 3.0
        input_data["heat_index_max"] = heat_idx_input + 1.5
        input_data["heat_index_p95"] = heat_idx_input
        input_data["extreme_heat_days_count"] = extreme_days
        input_data["pm25_mean"] = pm25_input
        input_data["pm25_p95"] = pm25_input * 1.2
        input_data["compound_risk_hi95_pm25"] = heat_idx_input * pm25_input
        input_data["compound_risk_himean_pm25"] = (heat_idx_input - 3.0) * pm25_input
        input_data["compound_risk_heatwave_pm25"] = extreme_days * (pm25_input * 1.2)
        input_data["uhi_thermal_amplification"] = (heat_idx_input + 1.5) * input_data["uhi_proxy_ratio"]
        input_data["compound_vulnerability_risk"] = input_data["compound_risk_hi95_pm25"] * input_data["sevi_vulnerability_index"]

        feat_cols = model_cardio.feature_name_
        X_sim = pd.DataFrame([input_data[feat_cols]])

        pred_cardio = float(models["Cardiorespiratory Total"].predict(X_sim)[0])
        pred_ihd = float(models["Ischemic Heart Disease"].predict(X_sim)[0])
        pred_hhd = float(models["Hypertensive Heart Disease"].predict(X_sim)[0])
        pred_asthma = float(models["Asthma Mortality"].predict(X_sim)[0])

        baseline_rate = float(df_master[df_master["adm3_en"] == selected_city]["rate_cardiorespiratory_per_100k"].mean())
        excess_pct = ((pred_cardio - baseline_rate) / (baseline_rate + 1e-5)) * 100.0

        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Cardio Rate", f"{pred_cardio:.2f} / 100k", f"{excess_pct:+.1f}% vs baseline")
        m2.metric("IHD Mortality Rate", f"{pred_ihd:.2f} / 100k")
        m3.metric("Hypertensive Rate", f"{pred_hhd:.2f} / 100k")
        m4.metric("Asthma Rate", f"{pred_asthma:.2f} / 100k")

        # Determine Alert Tier
        if heat_idx_input >= 41.0 or pm25_input >= 35.0:
            st.markdown("""
            <div class="alert-danger">
                <h4>🔴 ALERT LEVEL 4: SEVERE COMPOUND EMERGENCY</h4>
                <strong>Trigger:</strong> Extreme Heat Danger (>41°C) / Hazardous Fine Particulates (>35 µg/m³).<br>
                <strong>Mandatory Actions:</strong>
                <ul>
                    <li>Activate municipal emergency cooling centers in sports complexes and civic centers.</li>
                    <li>Trigger Hospital ER surge protocols (dedicated heat stroke and acute coronary triage).</li>
                    <li>Deploy mobile hydration and triage vans to high-SEVI informal settlements.</li>
                    <li>Enforce complete outdoor labor shutdown between 10:30 AM and 3:30 PM.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif heat_idx_input >= 37.0 or pm25_input >= 25.0 or extreme_days >= 3:
            st.markdown("""
            <div class="alert-warning">
                <h4>🟠 ALERT LEVEL 3: HIGH HAZARD WARNING</h4>
                <strong>Trigger:</strong> Heat Index 37°C–41°C / Unhealthy PM2.5 (25–35 µg/m³).<br>
                <strong>Mandatory Actions:</strong>
                <ul>
                    <li>Issue urgent hydration and cooling advisories via local radio and SMS cell broadcast.</li>
                    <li>Deploy CDRRMO urban misting cannons along high-traffic corridors.</li>
                    <li>Proactive barangay health worker check-ins on elderly and chronic COPD patients.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif heat_idx_input >= 32.0 or pm25_input >= 15.0:
            st.markdown("""
            <div class="alert-caution">
                <h4>🟡 ALERT LEVEL 2: CAUTION (TAG-INIT ADVISORY)</h4>
                <strong>Trigger:</strong> Moderate Thermal Caution (32°C–37°C) / Moderate PM2.5.<br>
                <strong>Mandatory Actions:</strong>
                <ul>
                    <li>Ensure Barangay Health Centers are fully stocked with bronchodilators and anti-hypertensives.</li>
                    <li>Mandate shaded rest breaks for traffic enforcers and street sweepers.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-normal">
                <h4>🟢 ALERT LEVEL 1: NORMAL / BASELINE</h4>
                <strong>Trigger:</strong> Normal baseline atmospheric and air quality conditions.<br>
                <strong>Actions:</strong> Routine public health monitoring and maintenance of green urban corridors.
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Municipal Spatial Vulnerability
# -----------------------------------------------------------------------------
elif app_mode == "🗺️ Municipal Spatial Vulnerability":
    st.markdown('<div class="main-header">🗺️ Municipal Spatial Vulnerability & UHI Profiling</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Urban Heat Island (UHI) Built-to-Green Ratio")
        uhi_df = df_master[["adm3_en", "uhi_proxy_ratio", "pct_builtup_mean", "pct_tree_cover_mean"]].drop_duplicates().sort_values("uhi_proxy_ratio", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=uhi_df, x="adm3_en", y="uhi_proxy_ratio", palette="magma", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("UHI Ratio: Built-Up / (Tree Cover + 0.01)")
        ax.set_xlabel("")
        st.pyplot(fig)

    with col2:
        st.subheader("Socio-Environmental Vulnerability Index (SEVI)")
        sevi_df = df_master[df_master["year"] == 2020][["adm3_en", "sevi_vulnerability_index", "rwi_mean", "city_pop_density_mean"]].drop_duplicates().sort_values("sevi_vulnerability_index", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=sevi_df, x="adm3_en", y="sevi_vulnerability_index", palette="mako", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("SEVI = (1 - RWI) × ln(1 + PopDensity)")
        ax.set_xlabel("")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# 3. Epidemiological GAM & SHAP Insights
# -----------------------------------------------------------------------------
elif app_mode == "🔬 Epidemiological GAM & SHAP Insights":
    st.markdown('<div class="main-header">🔬 Diagnostic GAM Splines & TreeSHAP Attributions</div>', unsafe_allow_html=True)
    
    st.subheader("1. Non-Linear Exposure-Response J-Curves (pygam)")
    st.image("output/figures/gam_exposure_response_splines.png", caption="GAM Spline Exposure-Response for Heat Index and PM2.5 (Holdout Evaluation)")

    st.subheader("2. Global TreeSHAP Feature Attribution & Compound Dependence")
    c1, c2 = st.columns(2)
    with c1:
        st.image("output/figures/shap_feature_importance_bar.png", caption="Global TreeSHAP Importance")
    with c2:
        st.image("output/figures/shap_compound_hazard_dependence.png", caption="Compound Hazard Interaction Dependence")

# -----------------------------------------------------------------------------
# 4. Model Benchmarks & Test Reports
# -----------------------------------------------------------------------------
elif app_mode == "📊 Model Benchmarks & Test Reports":
    st.markdown('<div class="main-header">📊 Model Benchmarks & Out-of-Time Test Reports</div>', unsafe_allow_html=True)
    
    st.subheader("Out-of-Time (2018–2021) Cross-Model Benchmark")
    df_metrics = pd.read_csv("output/model_evaluation_metrics.csv")
    st.dataframe(df_metrics, use_container_width=True)

    st.subheader("City-Disaggregated Performance Breakdown")
    df_city = pd.read_csv("output/city_disaggregated_metrics.csv")
    st.dataframe(df_city, use_container_width=True)

    st.subheader("Out-of-Time Forward Time Series Forecast (2018–2021)")
    st.image("output/figures/oot_time_series_forecast_comparison.png", caption="Actual vs. Predicted Cardiorespiratory Rates across Selected Flagship Cities")
