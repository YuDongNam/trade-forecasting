"""Data loading utilities for targets and exogenous features."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

from src.data.preprocess import (
    find_column,
    parse_date_column,
    aggregate_to_monthly,
    clean_numeric_column,
)
from src.config import Config, CaseConfig, ExogenousConfig


def load_target_series(
    csv_path: Path,
    date_col_candidates: List[str] = None,
    value_col_candidates: List[str] = None,
) -> pd.DataFrame:
    """
    Load and preprocess a target time series CSV.
    
    Args:
        csv_path: Path to CSV file
        date_col_candidates: List of candidate date column names
        value_col_candidates: List of candidate value column names
        
    Returns:
        DataFrame with columns ["ds", "y"] sorted by ds, monthly frequency
    """
    if date_col_candidates is None:
        date_col_candidates = ["ds", "Date", "date", "날짜", "observation_date"]
    
    if value_col_candidates is None:
        value_col_candidates = [
            "y",
            "PrimaryValue",
            "primaryValue",
            "value",
            "종가",
            "close",
            "Close",
        ]
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Find date and value columns
    date_col = find_column(df, date_col_candidates)
    value_col = find_column(df, value_col_candidates)
    
    # Parse dates
    df[date_col] = parse_date_column(df, date_col)
    
    # Clean value column (handle commas in numbers)
    df[value_col] = clean_numeric_column(df[value_col])
    
    # Aggregate to monthly
    monthly_df = aggregate_to_monthly(df, date_col, value_col, aggregation="sum")
    
    # Ensure ds is datetime
    monthly_df["ds"] = pd.to_datetime(monthly_df["ds"])
    
    # Remove rows with NaN y
    monthly_df = monthly_df[monthly_df["y"].notna()].copy()
    
    # Sort and remove duplicates
    monthly_df = monthly_df.sort_values("ds").reset_index(drop=True)
    monthly_df = monthly_df.drop_duplicates(subset=["ds"], keep="last")
    
    return monthly_df[["ds", "y"]]


def load_exogenous_feature(
    csv_path: Path,
    exo_config: ExogenousConfig,
    feature_name: str,
) -> pd.DataFrame:
    """
    Load and preprocess an exogenous feature CSV.
    
    Args:
        csv_path: Path to CSV file
        exo_config: Exogenous feature configuration
        feature_name: Name for the feature column in output
        
    Returns:
        DataFrame with columns ["ds", feature_name] sorted by ds, monthly frequency
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Find date and value columns
    date_col = find_column(df, exo_config.date_col_candidates)
    value_col = find_column(df, exo_config.value_col_candidates)
    
    # Parse dates
    df[date_col] = parse_date_column(df, date_col)
    
    # Clean value column
    df[value_col] = clean_numeric_column(df[value_col])
    
    # Aggregate to monthly
    monthly_df = aggregate_to_monthly(
        df, date_col, value_col, aggregation=exo_config.aggregation
    )
    
    # Rename y to feature name
    monthly_df = monthly_df.rename(columns={"y": feature_name})
    
    # Ensure ds is datetime
    monthly_df["ds"] = pd.to_datetime(monthly_df["ds"])
    
    # Remove rows with NaN
    monthly_df = monthly_df[monthly_df[feature_name].notna()].copy()
    
    # Sort and remove duplicates
    monthly_df = monthly_df.sort_values("ds").reset_index(drop=True)
    monthly_df = monthly_df.drop_duplicates(subset=["ds"], keep="last")
    
    return monthly_df[["ds", feature_name]]


def load_fx_rate(
    csv_path: Path,
    date_col_candidates: List[str] = None,
    value_col_candidates: List[str] = None,
) -> pd.DataFrame:
    """
    Load FX rate CSV (similar to target but different aggregation).
    
    Args:
        csv_path: Path to FX CSV file
        date_col_candidates: List of candidate date column names
        value_col_candidates: List of candidate value column names
        
    Returns:
        DataFrame with columns ["ds", "fx_close"] sorted by ds, monthly frequency
    """
    if date_col_candidates is None:
        date_col_candidates = ["ds", "Date", "date", "날짜"]
    
    if value_col_candidates is None:
        value_col_candidates = ["y", "value", "close", "Close", "종가"]
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Find date and value columns
    date_col = find_column(df, date_col_candidates)
    value_col = find_column(df, value_col_candidates)
    
    # Parse dates
    df[date_col] = parse_date_column(df, date_col)
    
    # Clean value column (handle commas)
    df[value_col] = clean_numeric_column(df[value_col])
    
    # Aggregate to monthly (last close of month)
    monthly_df = aggregate_to_monthly(df, date_col, value_col, aggregation="last")
    
    # Rename to fx_close
    monthly_df = monthly_df.rename(columns={"y": "fx_close"})
    
    # Ensure ds is datetime
    monthly_df["ds"] = pd.to_datetime(monthly_df["ds"])
    
    # Remove rows with NaN
    monthly_df = monthly_df[monthly_df["fx_close"].notna()].copy()
    
    # Sort and remove duplicates
    monthly_df = monthly_df.sort_values("ds").reset_index(drop=True)
    monthly_df = monthly_df.drop_duplicates(subset=["ds"], keep="last")
    
    return monthly_df[["ds", "fx_close"]]


def merge_target_with_exogenous(
    target_df: pd.DataFrame,
    exogenous_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge target series with exogenous features.
    
    Args:
        target_df: Target DataFrame with columns ["ds", "y"]
        exogenous_dfs: Dictionary of {feature_name: DataFrame} with columns ["ds", feature_name]
        
    Returns:
        Merged DataFrame with columns ["ds", "y", ...exogenous features]
    """
    merged = target_df.copy()
    
    # Merge each exogenous feature
    for feature_name, exo_df in exogenous_dfs.items():
        merged = merged.merge(
            exo_df,
            on="ds",
            how="left",
            suffixes=("", f"_{feature_name}"),
        )
    
    # Sort by date
    merged = merged.sort_values("ds").reset_index(drop=True)
    
    # Forward fill exogenous features
    exogenous_cols = [col for col in merged.columns if col not in ["ds", "y"]]
    for col in exogenous_cols:
        merged[col] = merged[col].ffill()
    
    # Keep only rows where y exists (target must be present)
    merged = merged[merged["y"].notna()].copy()
    
    return merged


def load_case_data(config: Config, case: CaseConfig) -> pd.DataFrame:
    """
    Load all data for a single case (target + exogenous features).
    
    Args:
        config: Main configuration
        case: Case configuration
        
    Returns:
        DataFrame with columns ["ds", "y", ...exogenous features]
    """
    data_dir = Path(config.data_dir)
    
    # Load target
    target_path = data_dir / case.target_csv
    target_df = load_target_series(target_path)
    
    # Load exogenous features
    exogenous_dfs = {}
    
    # Tariff
    tariff_path = data_dir / config.exogenous["tariff"].file
    tariff_df = load_exogenous_feature(
        tariff_path, config.exogenous["tariff"], "tariff_rate"
    )
    exogenous_dfs["tariff_rate"] = tariff_df
    
    # Oil
    oil_path = data_dir / config.exogenous["oil"].file
    oil_df = load_exogenous_feature(
        oil_path, config.exogenous["oil"], "oil_close"
    )
    exogenous_dfs["oil_close"] = oil_df
    
    # Copper
    copper_path = data_dir / config.exogenous["copper"].file
    copper_df = load_exogenous_feature(
        copper_path, config.exogenous["copper"], "copper_close"
    )
    exogenous_dfs["copper_close"] = copper_df
    
    # Fed rate
    fed_path = data_dir / config.exogenous["fed_rate"].file
    fed_df = load_exogenous_feature(
        fed_path, config.exogenous["fed_rate"], "fed_rate"
    )
    exogenous_dfs["fed_rate"] = fed_df
    
    # FX (if applicable)
    if case.uses_fx and case.fx_file:
        fx_path = data_dir / case.fx_file
        fx_df = load_fx_rate(fx_path)
        exogenous_dfs["fx_close"] = fx_df
    
    # Merge all
    merged_df = merge_target_with_exogenous(target_df, exogenous_dfs)
    
    return merged_df

