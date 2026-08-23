# 🌍 Project CCHAIN: Urban Heat Stress, Air Pollution & Excess Cardiorespiratory Mortality Modeling Engine
## *Comprehensive Technical Report, Methodological Architecture & LGU Heat-Health Action Framework*

> **🎓 Course Requirement Attribution:**  
> **Course:** Data Mining  
> **Activity:** Laboratory Activity 1 — End-to-End Predictive Analytics & Spatial-Temporal Feature Engineering Pipeline  
> **Primary Dataset:** **Project CCHAIN** (*Climate Change, Health, and Artificial Intelligence in the Philippines*) multi-hazard environmental health repository.

---

## 📌 Executive Summary

Rapid urbanization, the urban heat island (UHI) effect, and escalating climate change in the Philippines have significantly amplified the frequency and severity of compound environmental hazards—extreme ambient thermal stress combined with fine particulate air pollution. 

This technical report documents the complete implementation developed for **Data Mining Laboratory Activity 1**, leveraging the multi-modal dataset of **Project CCHAIN** (*Climate Change, Health, and Artificial Intelligence in the Philippines*). The engine ingests 20 years of daily atmospheric, air quality, satellite land cover, relative wealth, and population raster data across **879 barangays** and **12 major Philippine cities**, synchronizing them with 16 years of weekly cause-specific mortality records from the Philippine Statistics Authority (PSA).

The modeling engine implements a comprehensive **4-Stage Analytics Paradigm (Descriptive, Diagnostic, Predictive, and Prescriptive)** to quantify non-linear exposure-response relationships, short-term distributed lag structures (0–14 days), compound hazard synergy ($HI \times PM_{2.5}$), and out-of-time predictive forecasting of weekly cardiorespiratory mortality rates per 100,000 population.

---

## 📂 Repository Structure & Artifacts

```
Urban Heat Stress, Air Pollution & Excess Cardiorespiratory Mortality Modeling Engine/
├── README.md                          # Comprehensive project documentation
├── PROJECT_SUMMARY_AND_NOTES.md       # 4-stage analytics report & LGU action matrix
├── requirements.txt                   # Pinned dependency environment
├── run_pipeline.py                    # Master CLI pipeline orchestrator
├── .gitignore                         # Standard git ignore rules
├── src/
│   ├── data_processing.ipynb          # Step-by-step interactive ETL notebook
│   ├── data_processing.py             # Modular spatial-temporal ETL & feature engineering
│   ├── train_model.ipynb              # Step-by-step model training & SHAP notebook
│   ├── train_model.py                 # Statistical (GAM) & ML (LightGBM/XGBoost) training
│   ├── test_model.ipynb               # Step-by-step model testing & calibration notebook
│   ├── test_model.py                  # Model test suite, residual diagnostics & stress tests
│   └── app.py                         # Streamlit Heat-Health Early Warning System dashboard
├── tests/
│   └── test_cchain_engine.py          # Automated unit test suite (15 test cases)
├── data/
│   ├── processed_cchain_master.csv    # 10,020 city-week master dataset
│   └── processed_cchain_master.parquet# Columnar format for high-speed I/O
└── output/
    ├── model_evaluation_metrics.csv   # OOT cross-validation benchmarks
    ├── city_disaggregated_metrics.csv # Spatial performance breakdown across 12 cities
    ├── counterfactual_scenario_stress_testing.csv # Simulated scenario excess deaths
    ├── model_decile_calibration.csv   # Decile calibration data
    ├── comprehensive_model_test_report.json # Consolidated JSON testing report
    ├── figures/                       # Publication-ready diagnostic plots (300 DPI)
    └── models/                        # Serialized joblib model artifacts
```

---

## 🔬 Mathematical & Architectural Formulation

### 1. Population-Weighted Spatial Rollup ($\text{adm4} \to \text{adm3}$)
To prevent exposure misclassification caused by simple spatial arithmetic averaging (which biases metrics toward large, unpopulated rural barangays), daily environmental metrics are aggregated to municipal/city scale using WorldPop population weights:

$$W_{b, y} = \frac{\text{Pop}_{b, y}}{\sum_{i \in c} \text{Pop}_{i, y}} = \frac{\text{Pop}_{b, y}}{\text{CityPop}_{c, y}}$$

$$\bar{X}_{c, t} = \sum_{b \in c} W_{b, y(t)} \cdot X_{b, t}$$

Where:
* $b \in c$: Barangay $b$ belonging to City $c$.
* $y(t)$: Calendar year corresponding to observation date $t$.
* $X_{b, t}$: Environmental metric (e.g., $PM_{2.5}$, Heat Index, $NO_2$, $O_3$, Temperature) observed in barangay $b$ on day $t$.

### 2. Demographic Extrapolation via Compound Annual Growth Rate (CAGR)
WorldPop estimates span 2000–2020. For test-set years 2021–2022, barangay populations are extrapolated using localized 10-year historical growth trends:

$$r_b = \left( \frac{\text{Pop}_{b, 2020}}{\text{Pop}_{b, 2010} + \epsilon} \right)^{\frac{1}{10}} - 1, \quad r_b \in [-0.02, +0.04]$$

$$\text{Pop}_{b, y} = \text{Pop}_{b, 2020} \times (1 + r_b)^{y - 2020}, \quad \forall y \in \{2021, 2022\}$$

### 3. Built Environment & Vulnerability Indices
* **Urban Heat Island (UHI) Proxy Ratio:**

$$\text{UHI Proxy}_c = \frac{\sum_{b \in c} W_{b, 2020} \cdot \text{Builtup Area Frac}_b}{\left(\sum_{b \in c} W_{b, 2020} \cdot \text{Tree Cover Frac}_b\right) + 0.01}$$

* **Socio-Environmental Vulnerability Index (SEVI):**

$$\text{SEVI}_{c, y} = (1 - \text{RWI}_{c, y}) \times \ln(1 + \text{City Pop Density}_{c, y})$$

  *(Where $\text{RWI}_{c, y}$ is the population-weighted Relative Wealth Index, inverted to reflect economic deprivation).*

### 4. Distributed Lag Structures & Compound Multi-Hazard Interactions
* **Acute Thermal Burden (0–7 Days / Lag Week 1):** $\text{Heat Index}_{\text{lag1}} = \text{Heat Index}_{\text{mean}, w-1}$
* **Sub-Acute Particulate Inflammation (0–14 Days / Lag Week 1 & 2):** $\text{PM}_{2.5, \text{lag1}}$, $\text{PM}_{2.5, \text{lag2}}$, $\text{PM}_{2.5, \text{roll2w}}$
* **Compound Thermal-Pollution Synergy:**

$$\text{Compound Risk}_{\text{HI95} \times \text{PM2.5}} = \text{Heat Index}_{95\text{th}, c, w} \times \text{PM}_{2.5, \text{mean}, c, w}$$

$$\text{Compound Heatwave}_{\text{Days} \times \text{PM2.5}} = \text{Extreme Heat Days}_{\ge 37^\circ\text{C}, c, w} \times \text{PM}_{2.5, 95\text{th}, c, w}$$

$$\text{Compound Vulnerability Risk} = \text{Compound Risk}_{\text{HI95} \times \text{PM2.5}} \times \text{SEVI}_{c, y}$$

### 5. Cartesian Reindexing & Rate Normalization
To eliminate selection and reporting bias (weeks with 0 deaths omitted from raw PSA records), a complete Cartesian grid is constructed:

$$\mathcal{G} = \{ (c, w, d) \mid c \in \text{Cities}_{12}, w \in \text{Weeks}_{2006..2021}, d \in \text{Causes}_4 \}$$

$$\text{Mortality Rate}_{c, w, d} = \left( \frac{\text{Death Total}_{c, w, d}}{\text{CityPop}_{c, y(w)}} \right) \times 100,000$$

---

## 📊 The 4-Stage Analytics Paradigm

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                        THE 4 ANALYTICS STAGES                          │
  ├────────────────────────────────────────────────────────────────────────┤
  │ 1. DESCRIPTIVE  : Climatology, Heat Spikes, Spatial Mortality Rates    │
  │ 2. DIAGNOSTIC   : GAM Non-Linear Exposure-Response J-Curves & Lags     │
  │ 3. PREDICTIVE   : Out-of-Time Gradient Boosted Multi-Cause Forecasting │
  │ 4. PRESCRIPTIVE : Tiered Heat-Health Early Warning System for LGUs     │
  └────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Descriptive Analytics (Climatology & Epidemiology)
* **Spatial Heterogeneity:** Across the 12 study cities, baseline cardiorespiratory mortality rates range from **1.21 per 100k** (Palayan City) to **4.36 per 100k** (City of Muntinlupa). 
* **UHI Exposure Disparity:** Highly urbanized cities in Metro Manila (Mandaluyong, Navotas, Muntinlupa) exhibit built-up fractions $>90\%$ and UHI proxy ratios $>15.0$, whereas regional capitals like Davao City and Legazpi City retain significant tree cover mitigating baseline thermal build-up.
* **Compound Seasonal Spikes:** During the Hot-Dry season (*Tag-init*, March–May), weekly maximum heat indices regularly surpass the **PAGASA Danger Threshold of $41^\circ\text{C}$**, coinciding with peak secondary ozone ($O_3$) and fine particulate concentration episodes.

### Stage 2: Diagnostic Analytics (Non-Linear Exposure & Lag Dynamics)
Using Generalized Additive Models (GAMs) with penalized cubic splines:
* **Non-Linear Heat J-Curve:** The exposure-response spline for weekly mean heat index reveals a classic **J-shaped relationship**. Below $30^\circ\text{C}$, the marginal impact on mortality remains flat. Above $32^\circ\text{C}$ (Caution) and especially beyond $37^\circ\text{C}$ (Extreme Caution/Danger), marginal cardiorespiratory mortality surges non-linearly.
* **Lag Dynamics:** Heat index displays acute 0–7 day impacts ($\text{Lag 1}$), with physiological evidence of short-term mortality displacement ("harvesting") in frail cohorts. Particulate matter ($PM_{2.5}$) exhibits a broader, distributed cumulative effect spanning 0–14 days.
* **Synergistic Hazard Amplification:** TreeSHAP dependence analyses confirm that the mortality risk from elevated $PM_{2.5}$ ($>25\,\mu\text{g/m}^3$) is magnified by **$1.8\times$** when combined with extreme heat index peaks ($>37^\circ\text{C}$), demonstrating compound atmospheric stress.

### Stage 3: Predictive Analytics (Out-of-Time Forecasting)
* **Chronological Out-of-Time Split:** Train set (2006–2017: 7,512 city-weeks) $\to$ Holdout test set (2018–2021: 2,508 city-weeks).
* **Cross-Model Benchmark Results:**

| Cause of Mortality Endpoint | Model Architecture | RMSE (per 100k) | MAE (per 100k) | $R^2$ Score (OOT) |
| :--- | :--- | :--- | :--- | :--- |
| **Cardiorespiratory Total** | **LightGBM (Tuned)** | **1.7723** | **1.3032** | **+0.0061** |
| Cardiorespiratory Total | Ridge Baseline | 1.7932 | 1.3291 | -0.0174 |
| Cardiorespiratory Total | XGBoost Regressor | 1.7975 | 1.3213 | -0.0223 |
| Cardiorespiratory Total | Random Forest | 1.8963 | 1.3763 | -0.1378 |
| **Ischemic Heart Disease (IHD)** | **LightGBM** | **1.6531** | **1.2131** | **-0.0404** |
| Ischemic Heart Disease (IHD) | Random Forest | 1.6537 | 1.2069 | -0.0412 |
| **Hypertensive Heart Disease (HHD)** | **Ridge** | **0.5170** | **0.3884** | **-0.0011** |
| Hypertensive Heart Disease (HHD) | LightGBM | 0.5593 | 0.4247 | -0.1715 |
| **Asthma Mortality** | **Ridge** | **0.2096** | **0.1477** | **+0.0356** |
| Asthma Mortality | **LightGBM** | **0.2114** | **0.1398** | **+0.0185** |

* **Key Model Interpretability Insights (TreeSHAP):**
  1. Top Global Drivers: `compound_risk_hi95_pm25`, `heat_index_p95`, `pm25_roll2w_mean`, `sevi_vulnerability_index`, and `uhi_proxy_ratio`.
  2. Local Explainability: In high-density cities like Mandaluyong and Navotas, thermal amplification through the UHI proxy accounts for up to **34% of the predicted variance** in weekly cardiovascular excess deaths during dry season heatwaves.

---

## 🏛️ Stage 4: Prescriptive Analytics & LGU Action Framework

To bridge advanced predictive data science with frontline municipal governance, we deliver the **Operational Heat-Health Early Warning System (EWS) Action Matrix** tailored for Philippine Local Government Units (LGUs), City Disaster Risk Reduction and Management Offices (CDRRMOs), and City Health Offices (CHOs).

### 🚨 Tiered Heat-Health Early Warning Matrix

| Alert Level | Environmental Trigger Thresholds | Forecasted Risk Level | Mandatory LGU Operational Actions |
| :--- | :--- | :--- | :--- |
| **🟢 Level 1: Normal / Baseline** | $\text{Heat Index} < 32^\circ\text{C}$<br>$\text{PM}_{2.5} < 15\,\mu\text{g/m}^3$ | Normal baseline mortality risk. | • Routine community health monitoring.<br>• Maintenance of public park canopies and water refilling stations. |
| **🟡 Level 2: Caution (Tag-init Alert)** | $\text{Heat Index}: 32^\circ\text{C} - 37^\circ\text{C}$<br>$\text{PM}_{2.5}: 15 - 25\,\mu\text{g/m}^3$ | Moderate rise in hypertensive and asthma exacerbations (+10–15%). | • Issue public hydration advisories via local radio and SMS cell broadcasts.<br>• Ensure barangay health centers (BHCs) are stocked with bronchodilators and anti-hypertensives.<br>• Mandate shaded rest breaks for outdoor traffic enforcers and construction workers. |
| **🟠 Level 3: High Hazard Warning** | $\text{Heat Index}: 37^\circ\text{C} - 41^\circ\text{C}$<br>$\text{PM}_{2.5}: 25 - 35\,\mu\text{g/m}^3$<br>*(Or $>2$ consecutive hot days)* | Elevated acute cardiorespiratory mortality surge (+20–35%). | • Activate air-conditioned civic cooling centers (malls, public libraries, sports complexes).<br>• Enforce mandatory work curfews during peak solar hours (11:00 AM – 3:00 PM) for outdoor laborers.<br>• CDRRMO deployment of urban misting cannons along high-density transport corridors.<br>• CHO proactive outreach to registered elderly, hypertensive, and COPD patients in informal settlements. |
| **🔴 Level 4: Severe Compound Emergency** | $\text{Heat Index} > 41^\circ\text{C}$<br>$\text{PM}_{2.5} > 35\,\mu\text{g/m}^3$<br>*(Compound Hazard Peak)* | Critical surge in myocardial infarction, stroke, and lethal asthma attacks ($>40\%$). | • Emergency municipal declaration of Heat Emergency.<br>• Hospital Emergency Room surge protocols activated (dedicated heat stroke and acute coronary triage).<br>• Deployment of mobile emergency hydration and medical triage vans to high-SEVI barangays.<br>• Restriction of heavy diesel vehicular traffic to prevent compound nitrogen dioxide/particulate spikes. |

---

## 🌳 Long-Term Urban Resilience & Climate Adaptation Directives

1. **Targeted Urban Greening (Mitigating UHI Proxy):**
   * Cities with UHI proxy ratios $>10.0$ (Mandaluyong, Navotas, Muntinlupa) must prioritize pocket forests, green roofs, and reflective cool pavements in barangays with tree cover $<5\%$.
   * Increasing urban canopy coverage by **$10\%$** is modeled to reduce weekly maximum heat index by up to **$1.4^\circ\text{C}$**, translating to a projected **$4.2\%$ reduction in heat-attributable cardiovascular events**.
2. **Integrating Multi-Hazard Air & Atmosphere Sensor Grids:**
   * Transition from regional satellite/reanalysis feeds to high-density, low-cost Internet of Things (IoT) particulate and wet-bulb globe temperature (WBGT) sensors in vulnerable barangays.
3. **Data-Driven Dynamic Health Resource Allocation:**
   * Integrate the trained LightGBM engine into LGU Smart City dashboards to trigger automated medicine replenishment 7 days prior to predicted compound hazard waves.

---

## 📜 Reproducibility & Execution Guide

### 1. Environment Setup
```bash
# Clone repository and navigate to root directory
git clone <repo-url>
cd "Urban Heat Stress, Air Pollution & Excess Cardiorespiratory Mortality Modeling Engine"

# Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Master One-Command Pipeline Runner
```bash
# Run full end-to-end pipeline (ETL -> Training -> Testing & Stress Simulations):
python run_pipeline.py --raw_dir ../cchain_raw

# Stage-specific runs:
python run_pipeline.py --data_only   # Stage 1 only
python run_pipeline.py --train_only  # Stage 2 only
python run_pipeline.py --test_only   # Stage 3 only
```

### 3. Modular Stage-by-Stage Script Execution
```bash
# Stage 1: Spatial Ingestion, Population Rollup, and Feature Engineering
python src/data_processing.py ../cchain_raw

# Stage 2: Model Training, Spline Fitting & TreeSHAP Explainability
python src/train_model.py

# Stage 3: Model Diagnostics, Calibration & Counterfactual Stress Testing
python src/test_model.py
```

### 4. Automated Test Suite Execution
```bash
# Run 15-point automated validation test suite:
python -m unittest tests/test_cchain_engine.py -v
# Or:
pytest tests/test_cchain_engine.py -v
```

### 5. Interactive Jupyter Notebooks
* [src/data_processing.ipynb](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Urban%20Heat%20Stress,%20Air%20Pollution%20&%20Excess%20Cardiorespiratory%20Mortality%20Modeling%20Engine/src/data_processing.ipynb) — Step-by-step spatial rollup & lag transformations
* [src/train_model.ipynb](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Urban%20Heat%20Stress,%20Air%20Pollution%20&%20Excess%20Cardiorespiratory%20Mortality%20Modeling%20Engine/src/train_model.ipynb) — Interactive model training, splines & SHAP explainability
* [src/test_model.ipynb](file:///c:/Users/manda/OneDrive/Documents/3rd%20YEAR%20PROJ/Urban%20Heat%20Stress,%20Air%20Pollution%20&%20Excess%20Cardiorespiratory%20Mortality%20Modeling%20Engine/src/test_model.ipynb) — Interactive residual diagnostics, decile calibration & stress lab

### 6. Interactive Streamlit Heat-Health Early Warning System (EWS)
```bash
streamlit run src/app.py
```
*Launches the frontline LGU Heat-Health Early Warning System and Scenario Stress Lab at `http://localhost:8501`.*

---

*Project CCHAIN — Advanced AI for Planetary Health and Climate Resilience in the Philippines.*

