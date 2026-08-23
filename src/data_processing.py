"""
================================================================================
Urban Heat Stress, Air Pollution & Excess Cardiorespiratory Mortality Pipeline
Module: src/data_processing.py
--------------------------------------------------------------------------------
Spatial-temporal ETL & Feature Engineering Engine for Project CCHAIN:
1. Spatial Ingestion & Population-Weighted Rollup (Barangay adm4 -> City adm3)
2. Population Extrapolation via Compound Annual Growth Rate (CAGR) (2000-2022)
3. Zero-Padding & Cartesian Grid Alignment (eliminates PSA reporting bias)
4. Static Land Cover (ESA WorldCover) & Wealth Index (RWI) Integration
5. ISO Weekly Temporal Aggregation (mean, max, 95th percentile, hot days >37C)
6. Distributed 0-14 Day Lag Polynomials & Compound Multi-Hazard Interactions
7. Standardized Rate per 100,000 Host Population Calculation & Master Export
================================================================================
"""

import os
import sys
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CCHAIN_DataPipeline")


class CCHAINDataPipeline:
    """
    Production-grade Spatial-Temporal Ingestion & Feature Engineering Pipeline.
    """

    def __init__(self, raw_data_dir: Union[str, Path], output_dir: Union[str, Path] = "data"):
        """
        Initialize pipeline paths and verify data directories.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.raw_data_dir.exists():
            # Fallback path resolution
            candidate_paths = [
                Path("../cchain_raw"),
                Path("cchain_raw"),
                Path("data/cchain_raw"),
                Path("../../cchain_raw")
            ]
            for p in candidate_paths:
                if p.exists():
                    self.raw_data_dir = p
                    break

        logger.info(f"Initialized Pipeline with raw data directory: {self.raw_data_dir.resolve()}")
        logger.info(f"Output directory set to: {self.output_dir.resolve()}")

    def load_spatial_reference(self) -> pd.DataFrame:
        """
        Load location.csv spatial reference table (adm1 -> adm4 mapping).
        """
        fp = self.raw_data_dir / "location.csv"
        if not fp.exists():
            raise FileNotFoundError(f"location.csv not found at {fp}")
        
        df_loc = pd.read_csv(fp)
        logger.info(f"Loaded spatial reference: {len(df_loc)} barangays across {df_loc['adm3_pcode'].nunique()} cities.")
        return df_loc

    def build_population_weights(self, df_loc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load WorldPop population data, extrapolate 2021-2022 via CAGR, 
        and compute normalized population weights (W_b,y = Pop_b,y / CityPop_c,y).
        
        Returns:
            Tuple containing:
            1. df_weights: Barangay-level population weights per year (adm4_pcode, year, weight, pop_count_total, pop_density_mean)
            2. df_city_pop: City-level total population per year (adm3_pcode, year, city_pop_total, city_pop_density_mean)
        """
        fp = self.raw_data_dir / "worldpop_population.csv"
        if not fp.exists():
            raise FileNotFoundError(f"worldpop_population.csv not found at {fp}")
        
        logger.info("Ingesting WorldPop population raster aggregates...")
        df_pop = pd.read_csv(fp, usecols=["adm4_pcode", "date", "pop_count_total", "pop_density_mean"])
        df_pop["year"] = pd.to_datetime(df_pop["date"]).dt.year
        df_pop = df_pop.drop(columns=["date"])

        # Pivot to compute CAGR per barangay across 2010 to 2020
        piv = df_pop.pivot(index="adm4_pcode", columns="year", values="pop_count_total")
        piv_dens = df_pop.pivot(index="adm4_pcode", columns="year", values="pop_density_mean")

        if 2010 in piv.columns and 2020 in piv.columns:
            # Compound Annual Growth Rate = (Pop_2020 / Pop_2010)^(1/10) - 1
            # Clip between -2% and +4% annual growth to prevent extreme extrapolation anomalies
            cagr = ((piv[2020] / (piv[2010] + 1e-5)) ** (1.0 / 10.0) - 1.0).clip(-0.02, 0.04)
        else:
            cagr = pd.Series(0.012, index=piv.index) # Baseline 1.2% national growth rate

        # Extrapolate for 2021 and 2022
        pop_2020 = piv[2020] if 2020 in piv.columns else piv.iloc[:, -1]
        dens_2020 = piv_dens[2020] if 2020 in piv_dens.columns else piv_dens.iloc[:, -1]

        extrapolated_records = []
        for yr in [2021, 2022]:
            factor = (1.0 + cagr) ** (yr - 2020)
            extrap_pop = pop_2020 * factor
            extrap_dens = dens_2020 * factor
            for adm4, pop_val in extrap_pop.items():
                extrapolated_records.append({
                    "adm4_pcode": adm4,
                    "year": yr,
                    "pop_count_total": float(pop_val),
                    "pop_density_mean": float(extrap_dens.get(adm4, np.nan))
                })

        df_extrap = pd.DataFrame(extrapolated_records)
        df_pop_full = pd.concat([df_pop, df_extrap], ignore_index=True)

        # Merge with city reference
        df_pop_full = df_pop_full.merge(df_loc[["adm4_pcode", "adm3_pcode"]], on="adm4_pcode", how="inner")

        # Compute city total population per year
        city_totals = df_pop_full.groupby(["adm3_pcode", "year"]).agg(
            city_pop_total=("pop_count_total", "sum"),
            city_pop_density_mean=("pop_density_mean", "mean")
        ).reset_index()

        df_pop_full = df_pop_full.merge(city_totals, on=["adm3_pcode", "year"], how="left")
        df_pop_full["weight"] = df_pop_full["pop_count_total"] / (df_pop_full["city_pop_total"] + 1e-9)

        logger.info(f"Constructed population weights from {df_pop_full['year'].min()} to {df_pop_full['year'].max()}.")
        return df_pop_full[["adm4_pcode", "adm3_pcode", "year", "weight", "pop_count_total", "pop_density_mean"]], city_totals

    def build_static_and_socioeconomic_features(self, df_weights: pd.DataFrame, df_city_pop: pd.DataFrame) -> pd.DataFrame:
        """
        Compute City-level Urban Heat Island (UHI) proxies and Socio-Environmental Vulnerability Indices (SEVI).
        """
        logger.info("Engineering Built-Up / Tree Cover UHI proxies and RWI vulnerability indices...")
        
        # 1. Land Cover (ESA WorldCover)
        fp_esa = self.raw_data_dir / "esa_worldcover.csv"
        df_esa = pd.read_csv(fp_esa)
        
        # Normalize weights to year 2020 for static features
        weights_2020 = df_weights[df_weights["year"] == 2020][["adm4_pcode", "weight", "adm3_pcode"]]
        df_esa_merged = df_esa.merge(weights_2020, on="adm4_pcode", how="inner")
        
        # City-level population weighted landcover fractions
        df_esa_city = df_esa_merged.groupby("adm3_pcode").apply(
            lambda g: pd.Series({
                "pct_builtup_mean": np.sum(g["pct_area_builtup"] * g["weight"]),
                "pct_tree_cover_mean": np.sum(g["pct_area_tree_cover"] * g["weight"]),
                "pct_grassland_mean": np.sum(g["pct_area_grassland"] * g["weight"]),
                "pct_water_mean": np.sum(g["pct_area_permanent_water_bodies"] * g["weight"])
            }),
            include_groups=False
        ).reset_index()

        # UHI Proxy Ratio: Built-Up fraction divided by (Tree Cover + 0.01 epsilon)
        df_esa_city["uhi_proxy_ratio"] = df_esa_city["pct_builtup_mean"] / (df_esa_city["pct_tree_cover_mean"] + 0.01)

        # 2. Relative Wealth Index (RWI)
        fp_rwi = self.raw_data_dir / "tm_relative_wealth_index.csv"
        df_rwi = pd.read_csv(fp_rwi)
        df_rwi["year"] = pd.to_datetime(df_rwi["date"]).dt.year
        
        df_rwi_merged = df_rwi.merge(df_weights, on=["adm4_pcode", "year"], how="inner")
        df_rwi_city = df_rwi_merged.groupby(["adm3_pcode", "year"]).apply(
            lambda g: pd.Series({
                "rwi_mean": np.sum(g["rwi_mean"] * g["weight"])
            }),
            include_groups=False
        ).reset_index()

        # Expand RWI to full historical range (2006-2022) by backfilling baseline 2016 values
        all_cities = df_city_pop["adm3_pcode"].unique()
        all_years = df_city_pop["year"].unique()
        full_grid = pd.MultiIndex.from_product([all_cities, all_years], names=["adm3_pcode", "year"]).to_frame().reset_index(drop=True)
        
        df_rwi_full = full_grid.merge(df_rwi_city, on=["adm3_pcode", "year"], how="left")
        df_rwi_full["rwi_mean"] = df_rwi_full.groupby("adm3_pcode")["rwi_mean"].bfill().ffill()

        # 3. Merge with City Population Density and compute SEVI
        df_static = df_city_pop.merge(df_esa_city, on="adm3_pcode", how="left")
        df_static = df_static.merge(df_rwi_full, on=["adm3_pcode", "year"], how="left")

        # Socio-Environmental Vulnerability Index: (1 - RWI) * ln(1 + pop_density)
        df_static["sevi_vulnerability_index"] = (1.0 - df_static["rwi_mean"]) * np.log1p(df_static["city_pop_density_mean"])

        logger.info(f"Generated static and socioeconomic feature matrix for {len(df_static)} city-year pairs.")
        return df_static

    def rollup_daily_environmental_data(self, df_weights: pd.DataFrame) -> pd.DataFrame:
        """
        Stream and aggregate daily barangay-level air quality and atmospheric data to city-level 
        using population-weighted averaging.
        """
        logger.info("Starting memory-efficient streaming spatial rollup of climate and air quality...")

        weights_lookup = df_weights.set_index(["adm4_pcode", "year"])[["adm3_pcode", "weight"]].to_dict("index")

        # 1. Rollup Climate Atmosphere
        fp_atm = self.raw_data_dir / "climate_atmosphere.csv"
        atm_cols = ["adm4_pcode", "date", "tave", "tmin", "tmax", "heat_index", "rh", "wind_speed", "pr", "solar_rad", "uv_rad"]
        
        atm_agg_list = []
        chunk_size = 500000
        logger.info(f"Streaming atmosphere data from {fp_atm}...")
        for chunk in pd.read_csv(fp_atm, usecols=atm_cols, chunksize=chunk_size):
            chunk["year"] = pd.to_datetime(chunk["date"]).dt.year
            # Vectorized mapping
            mapped = chunk.set_index(["adm4_pcode", "year"]).index.map(weights_lookup.get)
            valid_mask = [m is not None for m in mapped]
            chunk = chunk[valid_mask].copy()
            mapped_valid = [m for m in mapped if m is not None]
            
            chunk["adm3_pcode"] = [m["adm3_pcode"] for m in mapped_valid]
            chunk["weight"] = [m["weight"] for m in mapped_valid]

            num_cols = ["tave", "tmin", "tmax", "heat_index", "rh", "wind_speed", "pr", "solar_rad", "uv_rad"]
            for col in num_cols:
                chunk[col] = chunk[col] * chunk["weight"]

            agg = chunk.groupby(["adm3_pcode", "date"])[num_cols].sum().reset_index()
            atm_agg_list.append(agg)

        df_atm_daily = pd.concat(atm_agg_list, ignore_index=True).groupby(["adm3_pcode", "date"]).sum().reset_index()
        logger.info(f"Completed atmosphere rollup: {len(df_atm_daily)} daily city records.")

        # 2. Rollup Climate Air Quality
        fp_air = self.raw_data_dir / "climate_air_quality.csv"
        air_cols = ["adm4_pcode", "date", "pm25", "pm10", "no2", "o3", "so2", "co"]
        
        air_agg_list = []
        logger.info(f"Streaming air quality data from {fp_air}...")
        for chunk in pd.read_csv(fp_air, usecols=air_cols, chunksize=chunk_size):
            chunk["year"] = pd.to_datetime(chunk["date"]).dt.year
            mapped = chunk.set_index(["adm4_pcode", "year"]).index.map(weights_lookup.get)
            valid_mask = [m is not None for m in mapped]
            chunk = chunk[valid_mask].copy()
            mapped_valid = [m for m in mapped if m is not None]
            
            chunk["adm3_pcode"] = [m["adm3_pcode"] for m in mapped_valid]
            chunk["weight"] = [m["weight"] for m in mapped_valid]

            num_cols = ["pm25", "pm10", "no2", "o3", "so2", "co"]
            for col in num_cols:
                chunk[col] = chunk[col].fillna(0.0) * chunk["weight"]

            agg = chunk.groupby(["adm3_pcode", "date"])[num_cols].sum().reset_index()
            air_agg_list.append(agg)

        df_air_daily = pd.concat(air_agg_list, ignore_index=True).groupby(["adm3_pcode", "date"]).sum().reset_index()
        logger.info(f"Completed air quality rollup: {len(df_air_daily)} daily city records.")

        # Merge daily atmosphere and air quality
        df_daily_env = df_atm_daily.merge(df_air_daily, on=["adm3_pcode", "date"], how="inner")
        logger.info(f"Fused daily environmental master: {len(df_daily_env)} records.")
        return df_daily_env

    def aggregate_weekly_features_and_lags(self, df_daily_env: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily environmental metrics to ISO weekly scale (mean, max, 95th percentile, hot days)
        and engineer 0-14 day distributed lag polynomials and compound interaction terms.
        """
        logger.info("Computing ISO weekly environmental aggregations and lag structures...")
        
        df = df_daily_env.copy()
        df["datetime"] = pd.to_datetime(df["date"])
        
        # ISO calendar mapping
        iso_cal = df["datetime"].dt.isocalendar()
        df["iso_year"] = iso_cal.year
        df["iso_week"] = iso_cal.week
        
        # Monday date anchor for the ISO week
        df["week_start_date"] = df["datetime"] - pd.to_timedelta(df["datetime"].dt.dayofweek, unit="D")

        # Weekly statistical aggregations
        def p95(x):
            return np.percentile(x.dropna(), 95) if len(x.dropna()) > 0 else np.nan

        weekly_agg = df.groupby(["adm3_pcode", "iso_year", "iso_week", "week_start_date"]).agg(
            heat_index_mean=("heat_index", "mean"),
            heat_index_max=("heat_index", "max"),
            heat_index_p95=("heat_index", p95),
            heat_index_std=("heat_index", "std"),
            extreme_heat_days_count=("heat_index", lambda x: np.sum(x >= 37.0)), # Days with Heat Index >= 37C
            tave_mean=("tave", "mean"),
            tave_max=("tave", "max"),
            tave_min=("tave", "min"),
            rh_mean=("rh", "mean"),
            wind_speed_mean=("wind_speed", "mean"),
            pr_sum=("pr", "sum"),
            solar_rad_mean=("solar_rad", "mean"),
            uv_rad_mean=("uv_rad", "mean"),
            pm25_mean=("pm25", "mean"),
            pm25_max=("pm25", "max"),
            pm25_p95=("pm25", p95),
            pm25_std=("pm25", "std"),
            pm10_mean=("pm10", "mean"),
            pm10_max=("pm10", "max"),
            no2_mean=("no2", "mean"),
            o3_mean=("o3", "mean"),
            so2_mean=("so2", "mean"),
            co_mean=("co", "mean")
        ).reset_index()

        weekly_agg = weekly_agg.sort_values(["adm3_pcode", "week_start_date"]).reset_index(drop=True)

        # Engineer 0-14 day distributed lag structures per city
        logger.info("Engineering distributed lag features (Lag-1w, Lag-2w, Rolling EWMA)...")
        lagged_dfs = []
        for adm3, grp in weekly_agg.groupby("adm3_pcode"):
            grp = grp.copy()
            # Heat index lags
            grp["heat_index_lag1"] = grp["heat_index_mean"].shift(1)
            grp["heat_index_lag2"] = grp["heat_index_mean"].shift(2)
            grp["heat_index_roll2w_mean"] = grp["heat_index_mean"].rolling(window=2, min_periods=1).mean()
            grp["heat_index_ewma"] = grp["heat_index_mean"].ewm(span=2, adjust=False).mean()

            # PM2.5 lags
            grp["pm25_lag1"] = grp["pm25_mean"].shift(1)
            grp["pm25_lag2"] = grp["pm25_mean"].shift(2)
            grp["pm25_roll2w_mean"] = grp["pm25_mean"].rolling(window=2, min_periods=1).mean()
            grp["pm25_ewma"] = grp["pm25_mean"].ewm(span=2, adjust=False).mean()

            # O3 and NO2 lags
            grp["o3_lag1"] = grp["o3_mean"].shift(1)
            grp["no2_lag1"] = grp["no2_mean"].shift(1)

            # Compound Hazard Interaction Terms
            grp["compound_risk_hi95_pm25"] = grp["heat_index_p95"] * grp["pm25_mean"]
            grp["compound_risk_himean_pm25"] = grp["heat_index_mean"] * grp["pm25_mean"]
            grp["compound_risk_heatwave_pm25"] = grp["extreme_heat_days_count"] * grp["pm25_p95"]
            grp["compound_risk_heat_o3"] = grp["heat_index_max"] * grp["o3_mean"]

            grp = grp.bfill().ffill()
            lagged_dfs.append(grp)

        df_weekly_features = pd.concat(lagged_dfs, ignore_index=True)

        logger.info(f"Constructed weekly feature matrix with shape {df_weekly_features.shape}.")
        return df_weekly_features

    def process_mortality_and_grid(self, df_city_pop: pd.DataFrame) -> pd.DataFrame:
        """
        Load PSA cause-specific weekly mortality data, construct a complete Cartesian product grid 
        (12 Cities x All ISO Weeks), zero-fill unrecorded weeks, and compute mortality rates per 100k.
        """
        logger.info("Processing PSA mortality totals and building zero-padded Cartesian grid...")
        
        fp_psa = self.raw_data_dir / "disease_psa_totals.csv"
        df_psa = pd.read_csv(fp_psa)
        df_psa["datetime"] = pd.to_datetime(df_psa["date"])
        # Map PSA reporting date to the nearest Monday week start
        df_psa["week_start_date"] = df_psa["datetime"] - pd.to_timedelta(df_psa["datetime"].dt.dayofweek, unit="D")

        # Filter cardiorespiratory target causes
        target_causes = {
            "ISCHEMIC HEART DISEASE": "death_ihd",
            "HYPERTENSIVE HEART DISEASE": "death_hhd",
            "ASTHMA": "death_asthma"
        }
        
        df_target = df_psa[df_psa["disease_common_name"].isin(target_causes.keys())].copy()
        df_target["cause_col"] = df_target["disease_common_name"].map(target_causes)

        # Aggregate total deaths per city, week, and cause
        piv_mort = df_target.pivot_table(
            index=["adm3_pcode", "week_start_date"],
            columns="cause_col",
            values="death_total",
            aggfunc="sum",
            fill_value=0
        ).reset_index()

        # Build full continuous Cartesian Grid across all 12 cities and all weeks from 2006 to 2021
        all_cities = df_psa["adm3_pcode"].unique()
        all_weeks = pd.date_range(start="2006-01-02", end="2021-12-27", freq="W-MON")
        cartesian_grid = pd.MultiIndex.from_product(
            [all_cities, all_weeks],
            names=["adm3_pcode", "week_start_date"]
        ).to_frame().reset_index(drop=True)

        # Merge and zero-fill missing weeks
        df_mort_grid = cartesian_grid.merge(piv_mort, on=["adm3_pcode", "week_start_date"], how="left")
        for c in ["death_ihd", "death_hhd", "death_asthma"]:
            if c not in df_mort_grid.columns:
                df_mort_grid[c] = 0
            df_mort_grid[c] = df_mort_grid[c].fillna(0)

        # Composite Cardiorespiratory Total
        df_mort_grid["death_cardiorespiratory_total"] = (
            df_mort_grid["death_ihd"] + df_mort_grid["death_hhd"] + df_mort_grid["death_asthma"]
        )

        # Merge with city population to compute rate per 100k
        df_mort_grid["year"] = df_mort_grid["week_start_date"].dt.year
        df_mort_grid = df_mort_grid.merge(df_city_pop[["adm3_pcode", "year", "city_pop_total"]], on=["adm3_pcode", "year"], how="left")

        # Standardize rates per 100,000 population
        pop_scale = df_mort_grid["city_pop_total"] / 100000.0
        df_mort_grid["rate_ihd_per_100k"] = df_mort_grid["death_ihd"] / pop_scale
        df_mort_grid["rate_hhd_per_100k"] = df_mort_grid["death_hhd"] / pop_scale
        df_mort_grid["rate_asthma_per_100k"] = df_mort_grid["death_asthma"] / pop_scale
        df_mort_grid["rate_cardiorespiratory_per_100k"] = df_mort_grid["death_cardiorespiratory_total"] / pop_scale

        logger.info(f"Constructed zero-padded mortality grid: {len(df_mort_grid)} city-week records.")
        return df_mort_grid

    def execute_pipeline(self) -> pd.DataFrame:
        """
        Execute end-to-end pipeline and export master modeling dataset.
        """
        logger.info("=== EXECUTING CCHAIN DATA PROCESSING PIPELINE ===")
        
        # 1. Spatial Reference
        df_loc = self.load_spatial_reference()
        
        # 2. Population Weights & Extrapolation
        df_weights, df_city_pop = self.build_population_weights(df_loc)
        
        # 3. Static Land Cover & Socioeconomic Features
        df_static = self.build_static_and_socioeconomic_features(df_weights, df_city_pop)
        
        # 4. Daily Environmental Spatial Rollup
        df_daily_env = self.rollup_daily_environmental_data(df_weights)
        
        # 5. Weekly Temporal Fusion & Lag Engineering
        df_weekly_env = self.aggregate_weekly_features_and_lags(df_daily_env)
        
        # 6. Mortality Zero-Padded Grid & Rate Calculation
        df_mort = self.process_mortality_and_grid(df_city_pop)
        
        # 7. Master Fusion
        logger.info("Fusing environmental features, static vulnerability indices, and mortality targets...")
        df_master = df_mort.merge(
            df_weekly_env,
            on=["adm3_pcode", "week_start_date"],
            how="inner"
        )
        
        df_master = df_master.merge(
            df_static.drop(columns=["city_pop_total"], errors="ignore"),
            on=["adm3_pcode", "year"],
            how="left"
        )

        # Merge city name for readability
        df_city_names = df_loc[["adm3_pcode", "adm3_en"]].drop_duplicates()
        df_master = df_master.merge(df_city_names, on="adm3_pcode", how="left")

        # 8. Calendar Harmonics & Seasonality
        df_master["month"] = df_master["week_start_date"].dt.month
        df_master["quarter"] = df_master["week_start_date"].dt.quarter
        df_master["sin_week"] = np.sin(2 * np.pi * df_master["iso_week"] / 52.1775)
        df_master["cos_week"] = np.cos(2 * np.pi * df_master["iso_week"] / 52.1775)

        # Philippine Climate Seasons: 
        # Hot-Dry (Tag-init: Mar-May), Wet (Tag-ulan: Jun-Nov), Cool-Dry (Tag-lamig: Dec-Feb)
        def assign_ph_season(m):
            if m in [3, 4, 5]:
                return "Hot-Dry"
            elif m in [6, 7, 8, 9, 10, 11]:
                return "Wet"
            else:
                return "Cool-Dry"
                
        df_master["ph_season"] = df_master["month"].apply(assign_ph_season)

        # Thermal UHI amplification term
        df_master["uhi_thermal_amplification"] = df_master["heat_index_max"] * df_master["uhi_proxy_ratio"]
        df_master["compound_vulnerability_risk"] = df_master["compound_risk_hi95_pm25"] * df_master["sevi_vulnerability_index"]

        # Final cleanup and sort
        df_master = df_master.sort_values(["adm3_pcode", "week_start_date"]).reset_index(drop=True)

        # 9. Export Artifacts
        csv_out = self.output_dir / "processed_cchain_master.csv"
        parquet_out = self.output_dir / "processed_cchain_master.parquet"
        
        logger.info(f"Exporting master dataset to {csv_out} and {parquet_out}...")
        df_master.to_csv(csv_out, index=False)
        df_master.to_parquet(parquet_out, index=False)

        logger.info(f"Pipeline executed successfully! Master dataset dimensions: {df_master.shape}")
        logger.info(f"Target cities: {df_master['adm3_en'].unique().tolist()}")
        logger.info(f"Date range: {df_master['week_start_date'].min().strftime('%Y-%m-%d')} to {df_master['week_start_date'].max().strftime('%Y-%m-%d')}")
        
        return df_master


if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "../cchain_raw"
    pipeline = CCHAINDataPipeline(raw_data_dir=raw_path, output_dir="data")
    pipeline.execute_pipeline()
