"""
================================================================================
Urban Heat Stress, Air Pollution & Excess Cardiorespiratory Mortality Engine
Master Orchestration Script: run_pipeline.py
--------------------------------------------------------------------------------
Executes the full end-to-end Project CCHAIN Pipeline:
1. Spatial Ingestion, Population Rollup, and Lag Feature Engineering (src/data_processing.py)
2. Dual Epidemiological GAM & Predictive Machine Learning Suite (src/train_model.py)
3. Model Diagnostics, Counterfactual Stress Testing, and Reporting (src/test_model.py)
================================================================================
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CCHAIN_MasterPipeline")


def run_full_cchain_pipeline(raw_dir: str = "../cchain_raw", run_data: bool = True, run_train: bool = True, run_test: bool = True):
    logger.info("================================================================================")
    logger.info("   PROJECT CCHAIN: URBAN HEAT STRESS & AIR POLLUTION MORTALITY MODELING ENGINE   ")
    logger.info("================================================================================")

    # 1. Data Processing
    if run_data:
        logger.info("\n>>> STAGE 1: EXECUTING SPATIAL-TEMPORAL ETL & FEATURE ENGINEERING PIPELINE <<<")
        from src.data_processing import CCHAINDataPipeline
        pipeline = CCHAINDataPipeline(raw_data_dir=raw_dir, output_dir="data")
        df_master = pipeline.execute_pipeline()
        logger.info(f"Stage 1 Complete. Master dataset shape: {df_master.shape}")

    # 2. Model Training & Explainability
    if run_train:
        logger.info("\n>>> STAGE 2: EXECUTING MODEL TRAINING, SPLINES & TREESHAP EXPLAINABILITY <<<")
        from src.train_model import CCHAINModelingEngine
        engine = CCHAINModelingEngine(data_path="data/processed_cchain_master.csv", output_dir="output")
        engine.run_full_training_pipeline()
        logger.info("Stage 2 Complete. Models and figures generated.")

    # 3. Model Testing & Diagnostics
    if run_test:
        logger.info("\n>>> STAGE 3: EXECUTING MODEL TESTING, RESIDUAL DIAGNOSTICS & STRESS TESTING <<<")
        from src.test_model import CCHAINModelTester
        tester = CCHAINModelTester(data_path="data/processed_cchain_master.csv", models_dir="output/models", output_dir="output")
        report = tester.run_all_tests()
        logger.info("Stage 3 Complete. Comprehensive test report compiled.")

    logger.info("\n================================================================================")
    logger.info("   ALL PIPELINE STAGES EXECUTED SUCCESSFULLY! SYSTEM IS FULLY OPERATIONAL.      ")
    logger.info("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project CCHAIN Master Pipeline Runner")
    parser.add_argument("--raw_dir", type=str, default="../cchain_raw", help="Path to raw CCHAIN dataset directory")
    parser.add_argument("--data_only", action="store_true", help="Run only the data processing stage")
    parser.add_argument("--train_only", action="store_true", help="Run only the model training stage")
    parser.add_argument("--test_only", action="store_true", help="Run only the model testing stage")

    args = parser.parse_args()

    if args.data_only:
        run_full_cchain_pipeline(raw_dir=args.raw_dir, run_data=True, run_train=False, run_test=False)
    elif args.train_only:
        run_full_cchain_pipeline(raw_dir=args.raw_dir, run_data=False, run_train=True, run_test=False)
    elif args.test_only:
        run_full_cchain_pipeline(raw_dir=args.raw_dir, run_data=False, run_train=False, run_test=True)
    else:
        run_full_cchain_pipeline(raw_dir=args.raw_dir, run_data=True, run_train=True, run_test=True)
