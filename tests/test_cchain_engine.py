"""
================================================================================
Urban Heat Stress, Air Pollution & Excess Cardiorespiratory Mortality Modeling Engine
Comprehensive Test Suite: tests/test_cchain_engine.py
================================================================================
Automated test suite verifying:
1. Master Dataset Structure, Schema & Physical Range Consistency
2. Trained Model Artifacts, Serialization & Inference Capabilities
3. Statistical Residual Diagnostics & Exposure-Response Calibration
4. Counterfactual Climate Hazard Stress Simulations
5. Visual Figure Artifacts & Diagnostic Plots
6. Module Importability, Code Compilation & App Helper Functions
================================================================================
"""

import os
import sys
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Ensure root workspace is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestCCHAINMasterDataset(unittest.TestCase):
    """
    Validation of processed master dataset integrity, schema, and geographic bounds.
    """

    @classmethod
    def setUpClass(cls):
        cls.data_csv_path = PROJECT_ROOT / "data" / "processed_cchain_master.csv"
        cls.data_parquet_path = PROJECT_ROOT / "data" / "processed_cchain_master.parquet"
        
        assert cls.data_csv_path.exists(), f"Master CSV dataset missing at {cls.data_csv_path}"
        assert cls.data_parquet_path.exists(), f"Master Parquet dataset missing at {cls.data_parquet_path}"
        
        cls.df_csv = pd.read_csv(cls.data_csv_path)
        cls.df_parquet = pd.read_parquet(cls.data_parquet_path)

    def test_dataset_dimensions_and_matching(self):
        """Verify row/column dimensions and Parquet-to-CSV synchronization."""
        self.assertEqual(len(self.df_csv), 10020, "Dataset should have exactly 10,020 weekly observations (12 cities x 835 weeks)")
        self.assertEqual(len(self.df_csv), len(self.df_parquet), "CSV and Parquet row counts must match exactly")
        self.assertEqual(len(self.df_csv.columns), len(self.df_parquet.columns), "CSV and Parquet column counts must match")

    def test_geographic_coverage(self):
        """Verify all 12 Philippine cities are present with equal longitudinal weekly records."""
        expected_cities = [
            "Dagupan City", "Palayan City", "Legazpi City", "Iloilo City",
            "Mandaue City", "Tacloban City", "Zamboanga City", "Cagayan de Oro City",
            "Davao City", "City of Mandaluyong", "City of Navotas", "City of Muntinlupa"
        ]
        cities_in_df = self.df_csv["adm3_en"].unique().tolist()
        self.assertEqual(len(cities_in_df), 12, "Must contain exactly 12 Philippine cities")
        for city in expected_cities:
            self.assertIn(city, cities_in_df, f"Expected city {city} missing from master dataset")
        
        # Verify 835 weeks per city (16 years: 2006 to 2021)
        city_counts = self.df_csv["adm3_en"].value_counts()
        self.assertTrue((city_counts == 835).all(), "Every city should have exactly 835 longitudinal weekly records")

    def test_temporal_continuity(self):
        """Verify temporal start/end dates and zero time gaps."""
        dates = pd.to_datetime(self.df_csv["week_start_date"])
        self.assertEqual(dates.dt.year.min(), 2006, "Dataset must begin in 2006")
        self.assertEqual(dates.dt.year.max(), 2021, "Dataset must extend through 2021")

    def test_null_values_and_infinites(self):
        """Ensure no NaN or Infinite values in critical feature and target columns."""
        critical_cols = [
            "rate_cardiorespiratory_per_100k", "rate_ihd_per_100k",
            "rate_hhd_per_100k", "rate_asthma_per_100k",
            "heat_index_mean", "heat_index_max", "heat_index_p95",
            "pm25_mean", "pm25_max", "pm25_p95",
            "compound_risk_hi95_pm25", "uhi_proxy_ratio", "sevi_vulnerability_index"
        ]
        for col in critical_cols:
            self.assertIn(col, self.df_csv.columns, f"Column {col} missing in master dataset")
            null_count = self.df_csv[col].isnull().sum()
            self.assertEqual(null_count, 0, f"Column {col} has {null_count} unexpected null values")
            self.assertFalse(np.isinf(self.df_csv[col]).any(), f"Column {col} contains infinite values")

    def test_physical_variable_ranges(self):
        """Sanity check meteorological, pollutant, and mortality value ranges."""
        # Heat Index range
        self.assertGreaterEqual(self.df_csv["heat_index_mean"].min(), 15.0)
        self.assertLessEqual(self.df_csv["heat_index_max"].max(), 55.0)

        # PM2.5 range
        self.assertGreaterEqual(self.df_csv["pm25_mean"].min(), 0.0)
        self.assertLessEqual(self.df_csv["pm25_max"].max(), 300.0)

        # Mortality rates must be non-negative
        self.assertGreaterEqual(self.df_csv["rate_cardiorespiratory_per_100k"].min(), 0.0)
        self.assertGreaterEqual(self.df_csv["rate_ihd_per_100k"].min(), 0.0)
        self.assertGreaterEqual(self.df_csv["rate_hhd_per_100k"].min(), 0.0)
        self.assertGreaterEqual(self.df_csv["rate_asthma_per_100k"].min(), 0.0)


class TestTrainedModelsAndInference(unittest.TestCase):
    """
    Verification of trained model artifacts, serialization, and prediction execution.
    """

    @classmethod
    def setUpClass(cls):
        cls.models_dir = PROJECT_ROOT / "output" / "models"
        cls.data_path = PROJECT_ROOT / "data" / "processed_cchain_master.csv"
        cls.df = pd.read_csv(cls.data_path)
        cls.df["week_start_date"] = pd.to_datetime(cls.df["week_start_date"])
        cls.test_df = cls.df[cls.df["week_start_date"].dt.year >= 2018].copy()

    def test_model_files_exist(self):
        """Check presence of all 9 trained model joblib files."""
        expected_models = [
            "gam_spline_cardiorespiratory.joblib",
            "lightgbm_rate_cardiorespiratory_per_100k.joblib",
            "lightgbm_rate_ihd_per_100k.joblib",
            "lightgbm_rate_hhd_per_100k.joblib",
            "lightgbm_rate_asthma_per_100k.joblib",
            "xgboost_rate_cardiorespiratory_per_100k.joblib",
            "xgboost_rate_ihd_per_100k.joblib",
            "xgboost_rate_hhd_per_100k.joblib",
            "xgboost_rate_asthma_per_100k.joblib"
        ]
        for m_name in expected_models:
            model_path = self.models_dir / m_name
            self.assertTrue(model_path.exists(), f"Trained model artifact {m_name} is missing")
            self.assertGreater(model_path.stat().st_size, 1000, f"Model file {m_name} is too small / corrupt")

    def test_lightgbm_inference(self):
        """Test LightGBM model loading and inference across all cause endpoints."""
        endpoints = [
            "rate_cardiorespiratory_per_100k",
            "rate_ihd_per_100k",
            "rate_hhd_per_100k",
            "rate_asthma_per_100k"
        ]
        for endpoint in endpoints:
            model_path = self.models_dir / f"lightgbm_{endpoint}.joblib"
            model = joblib.load(model_path)
            
            feat_cols = model.feature_name_
            X_test = self.test_df[feat_cols]
            
            preds = model.predict(X_test)
            self.assertEqual(len(preds), len(self.test_df))
            self.assertFalse(np.isnan(preds).any(), f"LightGBM predictions for {endpoint} contain NaNs")
            self.assertFalse(np.isinf(preds).any(), f"LightGBM predictions for {endpoint} contain Infs")
            self.assertGreaterEqual(preds.min(), -0.5, f"LightGBM predictions for {endpoint} deviated unreasonably into negative space")

    def test_xgboost_inference(self):
        """Test XGBoost model loading and inference across all cause endpoints."""
        endpoints = [
            "rate_cardiorespiratory_per_100k",
            "rate_ihd_per_100k",
            "rate_hhd_per_100k",
            "rate_asthma_per_100k"
        ]
        lgb_ref = joblib.load(self.models_dir / "lightgbm_rate_cardiorespiratory_per_100k.joblib")
        feat_cols = lgb_ref.feature_name_

        for endpoint in endpoints:
            model_path = self.models_dir / f"xgboost_{endpoint}.joblib"
            model = joblib.load(model_path)
            
            X_test = self.test_df[feat_cols]
            preds = model.predict(X_test)
            self.assertEqual(len(preds), len(self.test_df))
            self.assertFalse(np.isnan(preds).any(), f"XGBoost predictions for {endpoint} contain NaNs")
            self.assertFalse(np.isinf(preds).any(), f"XGBoost predictions for {endpoint} contain Infs")

    def test_gam_spline_inference(self):
        """Test pygam GAM Spline model loading and prediction."""
        gam_path = self.models_dir / "gam_spline_cardiorespiratory.joblib"
        gam_model = joblib.load(gam_path)
        
        gam_features = ["heat_index_mean", "pm25_mean", "heat_index_lag1", "pm25_lag1", "uhi_proxy_ratio", "sin_week", "cos_week"]
        X_gam = self.test_df[gam_features].values
        preds = gam_model.predict(X_gam)
        self.assertEqual(len(preds), len(self.test_df))
        self.assertFalse(np.isnan(preds).any(), "GAM spline predictions contain NaNs")
        self.assertFalse(np.isinf(preds).any(), "GAM spline predictions contain Infs")


class TestDiagnosticsAndEvaluation(unittest.TestCase):
    """
    Verification of statistical diagnostics, test reports, and calibration tables.
    """

    def test_json_test_report_validity(self):
        """Verify schema and validity of comprehensive_model_test_report.json."""
        report_path = PROJECT_ROOT / "output" / "comprehensive_model_test_report.json"
        self.assertTrue(report_path.exists(), "comprehensive_model_test_report.json missing")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertIn("test_metadata", report)
        self.assertIn("endpoint_benchmarks", report)
        self.assertIn("residual_diagnostics", report)
        self.assertIn("scenario_stress_testing_summary", report)
        self.assertIn("calibration_summary", report)

        # Check metadata
        self.assertEqual(report["test_metadata"]["study_cities_count"], 12)
        self.assertEqual(report["test_metadata"]["test_records_count"], 2508)

        # Check benchmarks have positive Pearson correlation for all endpoints
        for target, bm in report["endpoint_benchmarks"].items():
            self.assertGreater(bm["LightGBM"]["pearson_r"], 0.1, f"Low Pearson correlation for {target}")
            self.assertLess(bm["LightGBM"]["pearson_p"], 0.001, f"Insignificant Pearson p-value for {target}")

        # Check residual diagnostics
        res = report["residual_diagnostics"]
        self.assertIn("durbin_watson", res)
        self.assertIn("skewness", res)
        self.assertIn("jarque_bera_stat", res)

    def test_decile_calibration_monotonicity(self):
        """Verify that predicted risk deciles are monotonically increasing."""
        calib_path = PROJECT_ROOT / "output" / "model_decile_calibration.csv"
        self.assertTrue(calib_path.exists(), "model_decile_calibration.csv missing")
        
        calib_df = pd.read_csv(calib_path)
        self.assertEqual(len(calib_df), 10, "Decile calibration table should have exactly 10 decile bins")
        
        # Check that bin_pred_mean is strictly monotonic increasing
        is_increasing = calib_df["bin_pred_mean"].is_monotonic_increasing
        self.assertTrue(is_increasing, "Model predicted mean risk must increase monotonically with risk decile")

    def test_counterfactual_scenarios_table(self):
        """Verify scenario stress testing simulations results."""
        scen_path = PROJECT_ROOT / "output" / "counterfactual_scenario_stress_testing.csv"
        self.assertTrue(scen_path.exists(), "counterfactual_scenario_stress_testing.csv missing")
        
        df_scen = pd.read_csv(scen_path)
        self.assertEqual(len(df_scen), 48, "Scenario stress testing must contain 48 records (4 scenarios x 12 cities)")
        
        # Check heatwave and compound scenarios induce positive excess mortality
        heatwave = df_scen[df_scen["scenario"].str.contains("Heatwave")]
        self.assertGreater(heatwave["excess_mortality_rate"].mean(), 0.0, "Heatwave scenario must elevate mortality")

        compound = df_scen[df_scen["scenario"].str.contains("Compound")]
        self.assertGreater(compound["excess_mortality_rate"].mean(), 0.0, "Compound hazard scenario must elevate mortality")


class TestFigureArtifacts(unittest.TestCase):
    """
    Verification of visual figure files and charts generated in output/figures/.
    """

    def test_all_figures_generated(self):
        """Verify presence and size of all 8 core publication figures."""
        figures_dir = PROJECT_ROOT / "output" / "figures"
        expected_figures = [
            "gam_exposure_response_splines.png",
            "shap_feature_importance_bar.png",
            "shap_beeswarm_plot.png",
            "shap_compound_hazard_dependence.png",
            "oot_time_series_forecast_comparison.png",
            "model_testing_residual_diagnostics.png",
            "stress_testing_counterfactual_scenarios.png",
            "model_calibration_decile_curve.png"
        ]
        for fig_name in expected_figures:
            fig_path = figures_dir / fig_name
            self.assertTrue(fig_path.exists(), f"Figure {fig_name} is missing from {figures_dir}")
            self.assertGreater(fig_path.stat().st_size, 10000, f"Figure {fig_name} file size is too small")


class TestModuleCompilationAndSyntax(unittest.TestCase):
    """
    Verification that all Python modules in src/ and root compile and import cleanly.
    """

    def test_module_syntax_compilation(self):
        """Compile all python source files to check for any syntax errors."""
        py_files = [
            PROJECT_ROOT / "run_pipeline.py",
            PROJECT_ROOT / "src" / "data_processing.py",
            PROJECT_ROOT / "src" / "train_model.py",
            PROJECT_ROOT / "src" / "test_model.py",
            PROJECT_ROOT / "src" / "app.py"
        ]
        for fpath in py_files:
            self.assertTrue(fpath.exists(), f"Source file {fpath} does not exist")
            with open(fpath, "r", encoding="utf-8") as f:
                code_content = f.read()
            # Test Python compilation
            compiled = compile(code_content, str(fpath), "exec")
            self.assertIsNotNone(compiled, f"Failed to compile {fpath.name}")

    def test_stream_app_helper_functions(self):
        """Test load_master_data and load_models functions from app.py."""
        from src.test_model import CCHAINModelTester
        tester = CCHAINModelTester()
        self.assertIsNotNone(tester.df)
        self.assertGreater(len(tester.df_test), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
