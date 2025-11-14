"""Data preprocessing utilities."""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


def find_column(
    df: pd.DataFrame, candidates: List[str], default: Optional[str] = None
) -> str:
    """
    Find a column in DataFrame from a list of candidate names.
    
    Args:
        df: Input DataFrame
        candidates: List of candidate column names
        default: Default column name if none found
        
    Returns:
        Column name that exists in DataFrame
        
    Raises:
        ValueError: If no candidate column is found and no default provided
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    
    if default and default in df.columns:
        return default
    
    raise ValueError(
        f"Could not find column from candidates {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.Series:
    """
    Parse date column to pandas datetime.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        
    Returns:
        Series of datetime objects
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Failed to parse some dates in column {date_col}")
    return dates


def aggregate_to_monthly(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    aggregation: str = "last",
) -> pd.DataFrame:
    """
    Aggregate data to monthly frequency.
    
    Args:
        df: Input DataFrame with date and value columns
        date_col: Name of date column
        value_col: Name of value column
        aggregation: Aggregation method - "last", "mean", "sum", "max"
        
    Returns:
        DataFrame with monthly aggregated data, columns: ["ds", "y"]
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Convert to month start
    df["ds"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    
    # Aggregate by month
    if aggregation == "last":
        monthly = df.groupby("ds")[value_col].last().reset_index()
    elif aggregation == "mean":
        monthly = df.groupby("ds")[value_col].mean().reset_index()
    elif aggregation == "sum":
        monthly = df.groupby("ds")[value_col].sum().reset_index()
    elif aggregation == "max":
        monthly = df.groupby("ds")[value_col].max().reset_index()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    monthly.columns = ["ds", "y"]
    monthly = monthly.sort_values("ds").reset_index(drop=True)
    
    # Remove duplicates (keep last)
    monthly = monthly.drop_duplicates(subset=["ds"], keep="last")
    
    # Remove NaT
    monthly = monthly[monthly["ds"].notna()].copy()
    
    return monthly


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Clean numeric column by removing commas and converting to float.
    
    Args:
        series: Input series (may contain strings with commas)
        
    Returns:
        Series of floats
    """
    if series.dtype == "object":
        # Remove commas and convert
        cleaned = series.astype(str).str.replace(",", "", regex=False)
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")

