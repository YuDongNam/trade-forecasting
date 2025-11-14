"""Plotting utilities."""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict

from src.config import CaseConfig


def plot_case(
    case: CaseConfig,
    predictions_df: pd.DataFrame,
    metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """
    Create and save a plot for a single case.
    
    Args:
        case: Case configuration
        predictions_df: DataFrame with columns ["ds", "actual", "forecast"]
        metrics: Dictionary with RMSE, MAE, MAPE
        output_path: Path to save the plot
    """
    # Validate input data
    if predictions_df is None or len(predictions_df) == 0:
        raise ValueError(f"No data to plot for {case.id}")
    
    # Ensure ds is datetime
    predictions_df = predictions_df.copy()
    predictions_df["ds"] = pd.to_datetime(predictions_df["ds"])
    
    # Check for required columns
    required_cols = ["ds", "actual", "forecast"]
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for plotting {case.id}: {missing_cols}. "
            f"Available columns: {list(predictions_df.columns)}"
        )
    
    # Remove NaN values
    predictions_df = predictions_df.dropna(subset=["actual", "forecast"])
    
    if len(predictions_df) == 0:
        raise ValueError(f"All data points are NaN for {case.id}")
    
    # Sort by date
    predictions_df = predictions_df.sort_values("ds").reset_index(drop=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual
    ax.plot(
        predictions_df["ds"],
        predictions_df["actual"],
        "k-",
        label="Actual",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    
    # Plot forecast
    ax.plot(
        predictions_df["ds"],
        predictions_df["forecast"],
        "b-",
        label="Forecast",
        linewidth=2,
        alpha=0.7,
        marker="s",
        markersize=4,
    )
    
    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(
        f"{case.label}\n"
        f"RMSE: {metrics.get('RMSE', 0):.2f}, "
        f"MAE: {metrics.get('MAE', 0):.2f}, "
        f"MAPE: {metrics.get('MAPE', 0):.2f}%",
        fontsize=14,
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Plot saved to {output_path} ({len(predictions_df)} data points)")

