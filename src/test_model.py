"""
================================================================================
Urban Heat Stress, Air Pollution & Excess Cardiorespiratory Mortality Pipeline
Module: src/test_model.py
--------------------------------------------------------------------------------
Comprehensive Model Testing, Statistical Diagnostics & Stress Testing Suite:
1. Out-of-Time Model Validation across All Cause Endpoints
2. Residual Diagnostics (Normality, Heteroscedasticity, Autocorrelation DW)
3. Calibration Analysis across Exposure Deciles
4. Counterfactual Climate Hazard Stress Testing (Heatwave, Inversion, Compound)
5. City-Disaggregated Spatial Sensitivity Profiling
6. Generation of Diagnostic Figures and Comprehensive Testing Metrics
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
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CCHAIN_ModelTesting")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300


class CCHAINModelTester:
    """
    Comprehensive Testing & Diagnostic Engine for CCHAIN Epidemiological Models.
    """

    def __init__(self, data_path: str = "data/processed_cchain_master.csv", models_dir: str = "output/models", output_dir: str = "output"):
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(self.data_path)
        self.df["week_start_date"] = pd.to_datetime(self.df["week_start_date"])

        # Train/Test Split (OOT Split at 2018)
        self.df_train = self.df[self.df["week_start_date"].dt.year < 2018].copy()
        self.df_test = self.df[self.df["week_start_date"].dt.year >= 2018].copy()

        # Target definitions
        self.targets = [
            "rate_cardiorespiratory_per_100k",
            "rate_ihd_per_100k",
            "rate_hhd_per_100k",
            "rate_asthma_per_100k"
        ]

        # Load feature columns from lightgbm model feature_name_
        lgb_sample = joblib.load(self.models_dir / "lightgbm_rate_cardiorespiratory_per_100k.joblib")
        if hasattr(lgb_sample, "feature_name_"):
            self.feature_cols = list(lgb_sample.feature_name_)
        else:
            self.feature_cols = [c for c in self.df.columns if c not in self.targets + ["adm3_pcode", "adm3_en", "week_start_date", "ph_season"]]

        logger.info(f"Loaded {len(self.df_test)} test records (2018-2021) across {self.df_test['adm3_en'].nunique()} cities.")

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run test evaluation across all models and endpoints.
        """
        logger.info("=== STEP 1: COMPREHENSIVE OUT-OF-TIME BENCHMARKING ===")
        
        evaluation_results = {}
        
        for target in self.targets:
            lgb_path = self.models_dir / f"lightgbm_{target}.joblib"
            xgb_path = self.models_dir / f"xgboost_{target}.joblib"
            
            if not lgb_path.exists() or not xgb_path.exists():
                logger.warning(f"Model files for {target} not found in {self.models_dir}")
                continue

            lgb_model = joblib.load(lgb_path)
            xgb_model = joblib.load(xgb_path)

            X_test = self.df_test[self.feature_cols]
            y_test = self.df_test[target]

            preds_lgb = lgb_model.predict(X_test)
            preds_xgb = xgb_model.predict(X_test)

            # Compute detailed statistical metrics
            def calc_metrics(y_true, y_pred):
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                corr, pval = stats.pearsonr(y_true, y_pred)
                spearman_corr, sp_pval = stats.spearmanr(y_true, y_pred)
                med_ae = np.median(np.abs(y_true - y_pred))
                return {
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "median_ae": float(med_ae),
                    "r2": float(r2),
                    "pearson_r": float(corr),
                    "pearson_p": float(pval),
                    "spearman_rho": float(spearman_corr),
                    "spearman_p": float(sp_pval)
                }

            evaluation_results[target] = {
                "LightGBM": calc_metrics(y_test, preds_lgb),
                "XGBoost": calc_metrics(y_test, preds_xgb),
                "y_true_mean": float(y_test.mean()),
                "y_true_std": float(y_test.std()),
                "y_true_min": float(y_test.min()),
                "y_true_max": float(y_test.max()),
                "preds_lgb": preds_lgb,
                "preds_xgb": preds_xgb
            }

            logger.info(f"Target: {target:<30} | LightGBM R2: {evaluation_results[target]['LightGBM']['r2']:.4f} | Pearson r: {evaluation_results[target]['LightGBM']['pearson_r']:.4f} (p={evaluation_results[target]['LightGBM']['pearson_p']:.4e})")

        return evaluation_results

    def run_residual_diagnostics(self, evaluation_results: Dict[str, Any]):
        """
        Perform rigorous residual distribution diagnostics:
        - Skewness and Kurtosis (Normality)
        - Jarque-Bera Normality Test
        - Durbin-Watson Autocorrelation Test
        - Breusch-Pagan / Residual variance vs fitted (Homoscedasticity)
        """
        logger.info("=== STEP 2: RESIDUAL & STATISTICAL DIAGNOSTICS ===")

        target = "rate_cardiorespiratory_per_100k"
        y_test = self.df_test[target].values
        y_pred = evaluation_results[target]["preds_lgb"]
        residuals = y_test - y_pred

        # 1. Normality & Moments
        skew = stats.skew(residuals)
        kurt = stats.kurtosis(residuals)
        jb_stat, jb_p = stats.jarque_bera(residuals)

        # 2. Durbin-Watson Autocorrelation statistic: sum((e_t - e_{t-1})^2) / sum(e_t^2)
        diff_res = np.diff(residuals)
        dw_stat = np.sum(diff_res ** 2) / (np.sum(residuals ** 2) + 1e-9)

        # 3. Residual Correlation with Heat Index & PM2.5 (Checking for uncaptured linear/non-linear bias)
        corr_hi, p_hi = stats.pearsonr(self.df_test["heat_index_p95"], residuals)
        corr_pm, p_pm = stats.pearsonr(self.df_test["pm25_mean"], residuals)

        logger.info(f"Residual Mean: {np.mean(residuals):.4f} (Near 0 unbiased)")
        logger.info(f"Residual Std Dev: {np.std(residuals):.4f}")
        logger.info(f"Skewness: {skew:.4f} | Excess Kurtosis: {kurt:.4f} | Jarque-Bera p-val: {jb_p:.4e}")
        logger.info(f"Durbin-Watson Statistic: {dw_stat:.4f} (Values near 1.8-2.0 indicate minimal autocorrelation)")
        logger.info(f"Residual correlation with Heat Index p95: r={corr_hi:.4f} (p={p_hi:.4f})")
        logger.info(f"Residual correlation with PM2.5 mean: r={corr_pm:.4f} (p={p_pm:.4f})")

        # Plot Residual Diagnostics 4-Panel Figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel A: Actual vs Predicted with 45-degree reference line
        axes[0, 0].scatter(y_test, y_pred, alpha=0.35, color="#1f78b4", edgecolors="none", s=25)
        max_val = max(y_test.max(), y_pred.max())
        axes[0, 0].plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=2, label="Perfect 1:1 Agreement")
        # Regression trendline
        m, b = np.polyfit(y_test, y_pred, 1)
        axes[0, 0].plot(y_test, m * y_test + b, color="black", lw=1.5, label=f"Fit Line: y={m:.2f}x+{b:.2f}")
        axes[0, 0].set_title("A. Out-of-Time Actual vs. Predicted Mortality Rate", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Actual Mortality Rate (per 100k)")
        axes[0, 0].set_ylabel("Predicted Mortality Rate (per 100k)")
        axes[0, 0].legend(loc="upper left")

        # Panel B: Residuals vs Fitted Values (Homoscedasticity check)
        axes[0, 1].scatter(y_pred, residuals, alpha=0.35, color="#33a02c", edgecolors="none", s=25)
        axes[0, 1].axhline(0, color="red", linestyle="--", lw=2)
        axes[0, 1].set_title("B. Residuals vs. Fitted Values (Homoscedasticity)", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Fitted Values (Predicted Rate / 100k)")
        axes[0, 1].set_ylabel("Residuals (Actual - Predicted)")

        # Panel C: Residual Distribution & QQ-Plot / Histogram
        sns.histplot(residuals, kde=True, color="#e31a1c", ax=axes[1, 0], bins=35)
        axes[1, 0].axvline(0, color="black", linestyle="--", lw=1.5)
        axes[1, 0].set_title(f"C. Residual Error Distribution (Skew={skew:.2f}, Kurt={kurt:.2f})", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Residual Error")
        axes[1, 0].set_ylabel("Frequency")

        # Panel D: Residual Autocorrelation over Time
        axes[1, 1].plot(self.df_test["week_start_date"], residuals, color="#6a3d9a", lw=0.9, alpha=0.75)
        axes[1, 1].axhline(0, color="red", linestyle="--", lw=1.5)
        axes[1, 1].set_title(f"D. Longitudinal Residual Series (Durbin-Watson = {dw_stat:.2f})", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Holdout Test Date (2018–2021)")
        axes[1, 1].set_ylabel("Residual (Actual - Predicted)")

        plt.tight_layout()
        diag_path = self.figures_dir / "model_testing_residual_diagnostics.png"
        plt.savefig(diag_path, dpi=300)
        plt.close()
        logger.info(f"Saved residual diagnostics plot to {diag_path}")

        return {
            "skewness": float(skew),
            "excess_kurtosis": float(kurt),
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_p": float(jb_p),
            "durbin_watson": float(dw_stat),
            "corr_heat_index": float(corr_hi),
            "corr_pm25": float(corr_pm)
        }

    def run_counterfactual_stress_tests(self) -> pd.DataFrame:
        """
        Execute Counterfactual Climate Hazard Simulations:
        Simulates municipal excess mortality rate surge across 4 defined climate scenarios:
        1. Baseline Normal: Heat Index = 28°C, PM2.5 = 15 µg/m³
        2. Isolated Heatwave: Heat Index = 42°C (PAGASA Danger), PM2.5 = 15 µg/m³
        3. Isolated Stagnant PM2.5 Inversion: Heat Index = 28°C, PM2.5 = 55 µg/m³ (Severe Pollution)
        4. Compound Extreme Hazard: Heat Index = 42°C, PM2.5 = 55 µg/m³ + UHI Amplification
        """
        logger.info("=== STEP 3: COUNTERFACTUAL CLIMATE HAZARD STRESS TESTING ===")

        target = "rate_cardiorespiratory_per_100k"
        lgb_model = joblib.load(self.models_dir / f"lightgbm_{target}.joblib")

        # Base test template
        base_df = self.df_test.copy()

        scenarios = {
            "1. Baseline Normal (HI=28°C, PM2.5=15)": {
                "heat_index_mean": 28.0, "heat_index_max": 30.0, "heat_index_p95": 29.5,
                "extreme_heat_days_count": 0, "pm25_mean": 15.0, "pm25_max": 20.0, "pm25_p95": 18.0
            },
            "2. Isolated Heatwave (HI=42°C Danger)": {
                "heat_index_mean": 38.0, "heat_index_max": 42.0, "heat_index_p95": 41.5,
                "extreme_heat_days_count": 5, "pm25_mean": 15.0, "pm25_max": 20.0, "pm25_p95": 18.0
            },
            "3. Isolated PM2.5 Inversion (55 µg/m³)": {
                "heat_index_mean": 28.0, "heat_index_max": 30.0, "heat_index_p95": 29.5,
                "extreme_heat_days_count": 0, "pm25_mean": 55.0, "pm25_max": 75.0, "pm25_p95": 65.0
            },
            "4. Compound Extreme Hazard (HI=42°C + PM2.5=55)": {
                "heat_index_mean": 38.0, "heat_index_max": 42.0, "heat_index_p95": 41.5,
                "extreme_heat_days_count": 5, "pm25_mean": 55.0, "pm25_max": 75.0, "pm25_p95": 65.0
            }
        }

        scenario_records = []

        for scen_name, overrides in scenarios.items():
            scen_df = base_df.copy()
            for col, val in overrides.items():
                if col in scen_df.columns:
                    scen_df[col] = val

            # Recalculate compound interaction terms
            scen_df["compound_risk_hi95_pm25"] = scen_df["heat_index_p95"] * scen_df["pm25_mean"]
            scen_df["compound_risk_himean_pm25"] = scen_df["heat_index_mean"] * scen_df["pm25_mean"]
            scen_df["compound_risk_heatwave_pm25"] = scen_df["extreme_heat_days_count"] * scen_df["pm25_p95"]
            scen_df["uhi_thermal_amplification"] = scen_df["heat_index_max"] * scen_df["uhi_proxy_ratio"]
            scen_df["compound_vulnerability_risk"] = scen_df["compound_risk_hi95_pm25"] * scen_df["sevi_vulnerability_index"]

            # Forecast mortality
            preds = lgb_model.predict(scen_df[self.feature_cols])
            scen_df["simulated_rate"] = preds

            city_avg = scen_df.groupby(["adm3_pcode", "adm3_en"])["simulated_rate"].mean().reset_index()
            for _, r in city_avg.iterrows():
                scenario_records.append({
                    "scenario": scen_name,
                    "adm3_pcode": r["adm3_pcode"],
                    "city_name": r["adm3_en"],
                    "simulated_mortality_rate_per_100k": float(r["simulated_rate"])
                })

        df_scen = pd.DataFrame(scenario_records)

        # Compute relative percentage increase over baseline
        base_rates = df_scen[df_scen["scenario"].str.startswith("1. Baseline")].set_index("city_name")["simulated_mortality_rate_per_100k"].to_dict()
        df_scen["baseline_rate"] = df_scen["city_name"].map(base_rates)
        df_scen["excess_mortality_rate"] = df_scen["simulated_mortality_rate_per_100k"] - df_scen["baseline_rate"]
        df_scen["pct_excess_increase"] = (df_scen["excess_mortality_rate"] / (df_scen["baseline_rate"] + 1e-5)) * 100.0

        # Plot Scenario Stress Test Comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(
            data=df_scen,
            x="city_name",
            y="simulated_mortality_rate_per_100k",
            hue="scenario",
            palette=["#2ca25f", "#ff7f00", "#756bb1", "#de2d26"],
            ax=ax
        )
        ax.set_title("Counterfactual Climate Stress Testing: Weekly Cardiorespiratory Mortality Rate by Scenario", fontsize=13, fontweight="bold")
        ax.set_ylabel("Simulated Mortality Rate / 100,000 Host Pop")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Climate Stress Scenario", frameon=True, loc="upper right")
        plt.tight_layout()
        
        scen_fig_path = self.figures_dir / "stress_testing_counterfactual_scenarios.png"
        plt.savefig(scen_fig_path, dpi=300)
        plt.close()
        logger.info(f"Saved counterfactual stress test plot to {scen_fig_path}")

        df_scen.to_csv(self.output_dir / "counterfactual_scenario_stress_testing.csv", index=False)
        return df_scen

    def run_calibration_by_decile(self) -> pd.DataFrame:
        """
        Evaluate predictive calibration across deciles of predicted risk.
        """
        logger.info("=== STEP 4: DECILING CALIBRATION ANALYSIS ===")

        target = "rate_cardiorespiratory_per_100k"
        lgb_model = joblib.load(self.models_dir / f"lightgbm_{target}.joblib")

        test_df = self.df_test.copy()
        test_df["pred_rate"] = lgb_model.predict(test_df[self.feature_cols])

        test_df["risk_decile"] = pd.qcut(test_df["pred_rate"], q=10, labels=False, duplicates="drop") + 1
        
        calib_df = test_df.groupby("risk_decile").agg(
            bin_pred_mean=("pred_rate", "mean"),
            bin_actual_mean=(target, "mean"),
            sample_count=(target, "count")
        ).reset_index()

        calib_df["calibration_error"] = calib_df["bin_pred_mean"] - calib_df["bin_actual_mean"]

        # Plot Calibration Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(calib_df["bin_pred_mean"], calib_df["bin_actual_mean"], marker="o", color="#2b8cbe", lw=2, label="Model Calibration")
        
        min_c = min(calib_df["bin_pred_mean"].min(), calib_df["bin_actual_mean"].min())
        max_c = max(calib_df["bin_pred_mean"].max(), calib_df["bin_actual_mean"].max())
        ax.plot([min_c, max_c], [min_c, max_c], color="red", linestyle="--", label="Perfect Calibration")
        
        ax.set_title("Model Calibration Curve Across Predicted Risk Deciles", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Predicted Mortality Rate / 100k (Decile Bin)")
        ax.set_ylabel("Mean Observed Mortality Rate / 100k")
        ax.legend(loc="upper left")
        plt.tight_layout()

        calib_fig_path = self.figures_dir / "model_calibration_decile_curve.png"
        plt.savefig(calib_fig_path, dpi=300)
        plt.close()
        logger.info(f"Saved calibration curve to {calib_fig_path}")

        calib_df.to_csv(self.output_dir / "model_decile_calibration.csv", index=False)
        return calib_df

    def generate_test_report_summary(self, eval_results, resid_diag, stress_df, calib_df):
        """
        Compile all test metrics into a consolidated JSON test report.
        """
        report = {
            "test_metadata": {
                "test_period": "2018-01-01 to 2021-12-27",
                "test_records_count": len(self.df_test),
                "study_cities_count": self.df_test["adm3_en"].nunique(),
                "study_cities": self.df_test["adm3_en"].unique().tolist()
            },
            "endpoint_benchmarks": {
                target: {
                    "LightGBM": eval_results[target]["LightGBM"],
                    "XGBoost": eval_results[target]["XGBoost"],
                    "observed_distribution": {
                        "mean": eval_results[target]["y_true_mean"],
                        "std": eval_results[target]["y_true_std"],
                        "min": eval_results[target]["y_true_min"],
                        "max": eval_results[target]["y_true_max"]
                    }
                } for target in self.targets if target in eval_results
            },
            "residual_diagnostics": resid_diag,
            "scenario_stress_testing_summary": {
                scen: {
                    "mean_simulated_rate_per_100k": float(grp["simulated_mortality_rate_per_100k"].mean()),
                    "mean_excess_rate_per_100k": float(grp["excess_mortality_rate"].mean()),
                    "mean_pct_increase": float(grp["pct_excess_increase"].mean()),
                    "max_pct_increase_city": grp.loc[grp["pct_excess_increase"].idxmax()]["city_name"],
                    "max_pct_increase_val": float(grp["pct_excess_increase"].max())
                } for scen, grp in stress_df.groupby("scenario")
            },
            "calibration_summary": calib_df.to_dict(orient="records")
        }

        report_path = self.output_dir / "comprehensive_model_test_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported comprehensive test report to {report_path}")
        return report

    def run_all_tests(self):
        """
        Run end-to-end testing suite.
        """
        logger.info("=== STARTING COMPREHENSIVE MODEL TESTING SUITE ===")
        eval_results = self.run_comprehensive_evaluation()
        resid_diag = self.run_residual_diagnostics(eval_results)
        stress_df = self.run_counterfactual_stress_tests()
        calib_df = self.run_calibration_by_decile()
        report = self.generate_test_report_summary(eval_results, resid_diag, stress_df, calib_df)
        logger.info("=== MODEL TESTING SUITE COMPLETED SUCCESSFULLY! ===")
        return report


if __name__ == "__main__":
    tester = CCHAINModelTester()
    tester.run_all_tests()
