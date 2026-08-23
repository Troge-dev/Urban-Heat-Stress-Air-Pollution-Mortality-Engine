"""
================================================================================
Urban Heat Stress, Air Pollution & Excess Cardiorespiratory Mortality Pipeline
Module: src/train_model.py
--------------------------------------------------------------------------------
Dual Epidemiological (GAM/DLNM) and Machine Learning (LightGBM/XGBoost) Engine:
1. Chronological Out-of-Time (OOT) Train/Test Split (Train: 2006-2017, Test: 2018-2021)
2. Diagnostic GAM Spline Exposure-Response Curves (pygam) for Non-linear Dynamics
3. Predictive Regressors (LightGBM, XGBoost, Random Forest, Ridge) across Causes
4. Multi-Cause Forecasting (Cardiorespiratory Total, IHD, HHD, Asthma)
5. Comprehensive Evaluation (RMSE, MAE, R2, City-Disaggregated Metrics)
6. TreeSHAP Attribution & Compound Hazard Interaction Analysis
7. Automated Publication-Quality Figure & Model Artifact Generation
================================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Statistical & ML Libraries
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import shap

# pygam for non-linear exposure-response splines
try:
    from pygam import LinearGAM, s, f, te
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CCHAIN_ModelingEngine")

# Visualization styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300


class CCHAINModelingEngine:
    """
    Comprehensive Epidemiological and Predictive Machine Learning Modeling Engine.
    """

    def __init__(self, data_path: str = "data/processed_cchain_master.csv", output_dir: str = "output"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.models_dir = self.output_dir / "models"

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Modeling Engine with dataset: {self.data_path.resolve()}")
        logger.info(f"Outputs will be saved to: {self.output_dir.resolve()}")

        self.df = self.load_data()
        self.feature_cols = self.define_feature_columns()
        self.target_cols = [
            "rate_cardiorespiratory_per_100k",
            "rate_ihd_per_100k",
            "rate_hhd_per_100k",
            "rate_asthma_per_100k"
        ]

    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Processed master dataset not found at {self.data_path}")
        df = pd.read_csv(self.data_path)
        df["week_start_date"] = pd.to_datetime(df["week_start_date"])
        logger.info(f"Loaded master dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def define_feature_columns(self) -> List[str]:
        """
        Define candidate feature space for predictive modeling.
        """
        features = [
            # Thermal Stress Metrics
            "heat_index_mean", "heat_index_max", "heat_index_p95", "heat_index_std",
            "extreme_heat_days_count", "tave_mean", "tave_max", "tave_min",
            
            # Atmospheric & Moisture
            "rh_mean", "wind_speed_mean", "pr_sum", "solar_rad_mean", "uv_rad_mean",
            
            # Fine & Coarse Particulate Pollution
            "pm25_mean", "pm25_max", "pm25_p95", "pm25_std",
            "pm10_mean", "pm10_max",
            
            # Gaseous Pollutants
            "no2_mean", "o3_mean", "so2_mean", "co_mean",
            
            # Distributed Lag Structures (0-14 days)
            "heat_index_lag1", "heat_index_lag2", "heat_index_roll2w_mean", "heat_index_ewma",
            "pm25_lag1", "pm25_lag2", "pm25_roll2w_mean", "pm25_ewma",
            "o3_lag1", "no2_lag1",
            
            # Compound Hazard Interaction Terms
            "compound_risk_hi95_pm25", "compound_risk_himean_pm25",
            "compound_risk_heatwave_pm25", "compound_risk_heat_o3",
            "uhi_thermal_amplification", "compound_vulnerability_risk",
            
            # Built Environment & Socioeconomic Vulnerability
            "pct_builtup_mean", "pct_tree_cover_mean", "uhi_proxy_ratio",
            "rwi_mean", "city_pop_density_mean", "sevi_vulnerability_index",
            
            # Calendar Seasonality & Harmonics
            "month", "quarter", "sin_week", "cos_week"
        ]
        
        # Verify presence in dataframe
        available = [c for c in features if c in self.df.columns]
        logger.info(f"Selected {len(available)} features for modeling pipeline.")
        return available

    def split_out_of_time(self, split_year: int = 2018) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological Out-of-Time (OOT) Split:
        Train: 2006-01-02 to 2017-12-31 (~12 years)
        Test:  2018-01-01 to 2021-12-27 (~4 years)
        """
        df_train = self.df[self.df["week_start_date"].dt.year < split_year].copy()
        df_test = self.df[self.df["week_start_date"].dt.year >= split_year].copy()
        
        logger.info(f"OOT Split Applied at Year {split_year}:")
        logger.info(f" - Train shape: {df_train.shape[0]} records ({df_train['week_start_date'].min().date()} to {df_train['week_start_date'].max().date()})")
        logger.info(f" - Test shape:  {df_test.shape[0]} records ({df_test['week_start_date'].min().date()} to {df_test['week_start_date'].max().date()})")
        
        return df_train, df_test

    def fit_diagnostic_gam(self, df_train: pd.DataFrame, df_test: pd.DataFrame, target: str = "rate_cardiorespiratory_per_100k"):
        """
        Fit Generalized Additive Models (GAM) with penalized B-splines to uncover 
        non-linear exposure-response curves for Heat Index, PM2.5, and Compound Hazards.
        """
        if not PYGAM_AVAILABLE:
            logger.warning("pygam is not available. Skipping GAM exposure-response fitting.")
            return None

        logger.info(f"Fitting Diagnostic Generalized Additive Model (GAM) for target: {target}...")
        
        # Select key epidemiological predictors for GAM spline terms
        gam_features = ["heat_index_mean", "pm25_mean", "heat_index_lag1", "pm25_lag1", "uhi_proxy_ratio", "sin_week", "cos_week"]
        X_tr = df_train[gam_features].dropna()
        y_tr = df_train.loc[X_tr.index, target]

        X_te = df_test[gam_features].dropna()
        y_te = df_test.loc[X_te.index, target]

        # Build GAM model with smooth spline terms
        # s(0): heat_index_mean, s(1): pm25_mean, s(2): heat_index_lag1, s(3): pm25_lag1, s(4): uhi_proxy, s(5): sin_week, s(6): cos_week
        gam = LinearGAM(
            s(0, n_splines=15) + 
            s(1, n_splines=15) + 
            s(2, n_splines=12) + 
            s(3, n_splines=12) + 
            s(4, n_splines=8) + 
            s(5, n_splines=10) + 
            s(6, n_splines=10)
        )

        try:
            gam.gridsearch(X_tr.values, y_tr.values, progress=False)
            logger.info("GAM Spline Model fitted and optimized via Generalized Cross-Validation (GCV).")
            
            # Predict and evaluate
            preds_te = gam.predict(X_te.values)
            gam_r2 = r2_score(y_te, preds_te)
            gam_mae = mean_absolute_error(y_te, preds_te)
            gam_rmse = np.sqrt(mean_squared_error(y_te, preds_te))
            logger.info(f"GAM OOT Performance: R2={gam_r2:.4f}, MAE={gam_mae:.4f}, RMSE={gam_rmse:.4f}")

            # Plot Non-Linear Exposure-Response Curves
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # 1. Heat Index Exposure-Response
            XX = gam.generate_X_grid(term=0)
            pdep, confi = gam.partial_dependence(term=0, X=XX, width=0.95)
            axes[0].plot(XX[:, 0], pdep, color="#d95f02", lw=2.5, label="Spline Response")
            axes[0].fill_between(XX[:, 0], confi[:, 0], confi[:, 1], color="#d95f02", alpha=0.2, label="95% CI")
            axes[0].axvline(37.0, color="red", linestyle="--", alpha=0.7, label="PAGASA Heat Danger (37°C)")
            axes[0].set_title("Non-Linear Heat Index Response Curve (J-Shape)", fontsize=12, fontweight="bold")
            axes[0].set_xlabel("Weekly Mean Heat Index (°C)")
            axes[0].set_ylabel("Marginal Impact on Mortality Rate / 100k")
            axes[0].legend(loc="upper left")

            # 2. PM2.5 Exposure-Response
            XX_pm = gam.generate_X_grid(term=1)
            pdep_pm, confi_pm = gam.partial_dependence(term=1, X=XX_pm, width=0.95)
            axes[1].plot(XX_pm[:, 1], pdep_pm, color="#7570b3", lw=2.5, label="Spline Response")
            axes[1].fill_between(XX_pm[:, 1], confi_pm[:, 0], confi_pm[:, 1], color="#7570b3", alpha=0.2, label="95% CI")
            axes[1].axvline(25.0, color="darkred", linestyle="--", alpha=0.7, label="WHO PM2.5 Guideline (25 µg/m³)")
            axes[1].set_title("Non-Linear PM2.5 Inhalation Response Curve", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Weekly Mean PM2.5 (µg/m³)")
            axes[1].set_ylabel("Marginal Impact on Mortality Rate / 100k")
            axes[1].legend(loc="upper left")

            # 3. Lag-1 Week Heat Index Response
            XX_lag = gam.generate_X_grid(term=2)
            pdep_lag, confi_lag = gam.partial_dependence(term=2, X=XX_lag, width=0.95)
            axes[2].plot(XX_lag[:, 2], pdep_lag, color="#e7298a", lw=2.5, label="Lag-1w Response")
            axes[2].fill_between(XX_lag[:, 2], confi_lag[:, 0], confi_lag[:, 1], color="#e7298a", alpha=0.2, label="95% CI")
            axes[2].set_title("Delayed Thermal Stress (Lag-1 Week Dynamic)", fontsize=12, fontweight="bold")
            axes[2].set_xlabel("Lag-1 Week Heat Index (°C)")
            axes[2].set_ylabel("Marginal Impact on Mortality Rate / 100k")
            axes[2].legend(loc="upper left")

            plt.tight_layout()
            gam_fig_path = self.figures_dir / "gam_exposure_response_splines.png"
            plt.savefig(gam_fig_path, dpi=300)
            plt.close()
            logger.info(f"Saved GAM exposure-response curves to {gam_fig_path}")

            joblib.dump(gam, self.models_dir / "gam_spline_cardiorespiratory.joblib")
            return gam
        except Exception as e:
            logger.error(f"Error fitting GAM: {e}")
            return None

    def train_predictive_models(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Train and benchmark LightGBM, XGBoost, Random Forest, and Ridge Regressors 
        across cause-specific mortality endpoints.
        """
        logger.info("=== TRAINING PREDICTIVE MACHINE LEARNING SUITE ===")
        
        results = {}
        all_metrics = []

        X_train = df_train[self.feature_cols]
        X_test = df_test[self.feature_cols]

        for target in self.target_cols:
            logger.info(f"\n--- Training Models for Endpoint: {target} ---")
            y_train = df_train[target]
            y_test = df_test[target]

            target_models = {}

            # 1. LightGBM Regressor (Tuned for epidemiological tabular time series)
            lgb_model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_preds = lgb_model.predict(X_test)
            target_models["LightGBM"] = (lgb_model, lgb_preds)

            # 2. XGBoost Regressor
            xgb_model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            xgb_preds = xgb_model.predict(X_test)
            target_models["XGBoost"] = (xgb_model, xgb_preds)

            # 3. Random Forest Regressor
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            rf_preds = rf_model.predict(X_test)
            target_models["RandomForest"] = (rf_model, rf_preds)

            # 4. Ridge Baseline Regressor
            ridge_model = Ridge(alpha=10.0, random_state=42)
            ridge_model.fit(X_train.fillna(0), y_train)
            ridge_preds = ridge_model.predict(X_test.fillna(0))
            target_models["Ridge"] = (ridge_model, ridge_preds)

            # Evaluate all models for this target
            for name, (model, preds) in target_models.items():
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                
                logger.info(f"Model: {name:<14} | Target: {target:<30} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
                
                all_metrics.append({
                    "target": target,
                    "model": name,
                    "rmse": round(rmse, 4),
                    "mae": round(mae, 4),
                    "r2": round(r2, 4)
                })

                # Save best LightGBM and XGBoost model artifacts
                if name in ["LightGBM", "XGBoost"]:
                    joblib.dump(model, self.models_dir / f"{name.lower()}_{target}.joblib")

            results[target] = target_models

        # Save metrics table
        df_metrics = pd.DataFrame(all_metrics)
        metrics_csv_path = self.output_dir / "model_evaluation_metrics.csv"
        df_metrics.to_csv(metrics_csv_path, index=False)
        logger.info(f"\nSaved cross-model performance metrics to {metrics_csv_path}")

        return results

    def compute_city_disaggregated_performance(self, df_test: pd.DataFrame, results: Dict[str, Any]):
        """
        Evaluate model performance on a city-by-city basis to assess spatial robustness.
        """
        logger.info("Computing city-level disaggregated evaluation metrics...")
        
        target = "rate_cardiorespiratory_per_100k"
        best_model, _ = results[target]["LightGBM"]
        
        test_df = df_test.copy()
        test_df["predicted_rate"] = best_model.predict(test_df[self.feature_cols])

        city_metrics = []
        for city_pcode, grp in test_df.groupby("adm3_pcode"):
            city_name = grp["adm3_en"].iloc[0]
            rmse = np.sqrt(mean_squared_error(grp[target], grp["predicted_rate"]))
            mae = mean_absolute_error(grp[target], grp["predicted_rate"])
            r2 = r2_score(grp[target], grp["predicted_rate"]) if len(grp) > 2 else 0.0
            
            city_metrics.append({
                "adm3_pcode": city_pcode,
                "city_name": city_name,
                "mean_actual_rate": round(grp[target].mean(), 3),
                "mean_pred_rate": round(grp["predicted_rate"].mean(), 3),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4)
            })

        df_city_metrics = pd.DataFrame(city_metrics).sort_values("r2", ascending=False)
        city_metrics_csv = self.output_dir / "city_disaggregated_metrics.csv"
        df_city_metrics.to_csv(city_metrics_csv, index=False)
        logger.info(f"Saved city-disaggregated metrics to {city_metrics_csv}")
        return df_city_metrics

    def explain_models_with_shap(self, df_train: pd.DataFrame, df_test: pd.DataFrame, results: Dict[str, Any]):
        """
        Apply TreeSHAP to quantify global feature importances, isolate thermal vs. pollution impacts,
        and visualize compound interaction effects.
        """
        logger.info("=== COMPUTING TREESHAP INTERPRETABILITY ENGINE ===")
        
        target = "rate_cardiorespiratory_per_100k"
        lgb_model, _ = results[target]["LightGBM"]

        X_train = df_train[self.feature_cols]
        X_test = df_test[self.feature_cols]

        # Initialize TreeExplainer
        explainer = shap.TreeExplainer(lgb_model)
        # Sample background and test subset for fast, robust SHAP calculation
        sample_test = X_test.sample(min(len(X_test), 1000), random_state=42)
        shap_values = explainer(sample_test)

        logger.info("Generated TreeSHAP attributions for holdout test sample.")

        # 1. SHAP Global Feature Importance Bar Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title("Global TreeSHAP Feature Attribution (Cardiorespiratory Mortality)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        shap_bar_path = self.figures_dir / "shap_feature_importance_bar.png"
        plt.savefig(shap_bar_path, dpi=300)
        plt.close()
        logger.info(f"Saved SHAP bar plot to {shap_bar_path}")

        # 2. SHAP Beeswarm Plot (Directional Impact)
        fig, ax = plt.subplots(figsize=(11, 8))
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.title("TreeSHAP Beeswarm: Directional Driver of Mortality Surges", fontsize=13, fontweight="bold")
        plt.tight_layout()
        shap_beeswarm_path = self.figures_dir / "shap_beeswarm_plot.png"
        plt.savefig(shap_beeswarm_path, dpi=300)
        plt.close()
        logger.info(f"Saved SHAP beeswarm plot to {shap_beeswarm_path}")

        # 3. SHAP Dependence Plot: Heat Index 95th Percentile vs PM2.5 Mean
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.plots.scatter(
            shap_values[:, "compound_risk_hi95_pm25"],
            color=shap_values[:, "heat_index_p95"],
            show=False
        )
        plt.title("Compound Multi-Hazard Synergy: (Heat Index 95th × PM2.5 Mean)", fontsize=12, fontweight="bold")
        plt.xlabel("Compound Risk Index (HeatIndex_p95 × PM2.5_mean)")
        plt.ylabel("SHAP Value (Impact on Mortality Rate / 100k)")
        plt.tight_layout()
        shap_dep_path = self.figures_dir / "shap_compound_hazard_dependence.png"
        plt.savefig(shap_dep_path, dpi=300)
        plt.close()
        logger.info(f"Saved SHAP compound dependence plot to {shap_dep_path}")

    def plot_time_series_forecast(self, df_test: pd.DataFrame, results: Dict[str, Any]):
        """
        Plot out-of-time actual vs. predicted weekly mortality rates for selected flagship cities.
        """
        logger.info("Generating Out-of-Time Actual vs. Predicted time-series comparisons...")
        
        target = "rate_cardiorespiratory_per_100k"
        best_model, _ = results[target]["LightGBM"]

        test_df = df_test.copy()
        test_df["predicted_rate"] = best_model.predict(test_df[self.feature_cols])

        flagship_cities = ["Davao City", "City of Mandaluyong", "Iloilo City", "Cagayan de Oro City"]
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
        axes = axes.flatten()

        for i, city in enumerate(flagship_cities):
            city_data = test_df[test_df["adm3_en"] == city].sort_values("week_start_date")
            if len(city_data) == 0:
                continue

            ax = axes[i]
            ax.plot(city_data["week_start_date"], city_data[target], label="Actual Mortality Rate", color="#2b8cbe", lw=1.8)
            ax.plot(city_data["week_start_date"], city_data["predicted_rate"], label="LightGBM Forecast", color="#e41a1c", linestyle="--", lw=1.8)
            
            # Highlight heatwave peaks (>37C)
            hw_weeks = city_data[city_data["extreme_heat_days_count"] > 0]
            if len(hw_weeks) > 0:
                ax.scatter(hw_weeks["week_start_date"], hw_weeks[target], color="orange", s=30, zorder=5, label="Extreme Heat Week (>=37°C)")

            ax.set_title(f"Holdout Out-of-Time Forecast: {city} (2018–2021)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Cardiorespiratory Rate / 100k")
            ax.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        ts_fig_path = self.figures_dir / "oot_time_series_forecast_comparison.png"
        plt.savefig(ts_fig_path, dpi=300)
        plt.close()
        logger.info(f"Saved time-series forecast plot to {ts_fig_path}")

    def run_full_training_pipeline(self):
        """
        Execute full training, evaluation, explainability, and visualization workflow.
        """
        logger.info("=== STARTING CCHAIN MODEL TRAINING & EVALUATION WORKFLOW ===")
        
        # 1. Chronological Split
        df_train, df_test = self.split_out_of_time(split_year=2018)

        # 2. Fit Diagnostic GAM Splines
        gam_model = self.fit_diagnostic_gam(df_train, df_test, target="rate_cardiorespiratory_per_100k")

        # 3. Train Predictive ML Suite
        results = self.train_predictive_models(df_train, df_test)

        # 4. City-Level Disaggregated Performance
        self.compute_city_disaggregated_performance(df_test, results)

        # 5. TreeSHAP Interpretability
        self.explain_models_with_shap(df_train, df_test, results)

        # 6. Out-of-Time Visual Forecast
        self.plot_time_series_forecast(df_test, results)

        logger.info("=== MODELING ENGINE EXECUTION COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    engine = CCHAINModelingEngine(
        data_path="data/processed_cchain_master.csv",
        output_dir="output"
    )
    engine.run_full_training_pipeline()
