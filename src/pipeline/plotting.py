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
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual
    ax.plot(
        predictions_df["ds"],
        predictions_df["actual"],
        "k-",
        label="Actual",
        linewidth=2,
    )
    
    # Plot forecast
    ax.plot(
        predictions_df["ds"],
        predictions_df["forecast"],
        "b-",
        label="Forecast",
        linewidth=2,
        alpha=0.7,
    )
    
    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(
        f"{case.label}\n"
        f"RMSE: {metrics['RMSE']:.2f}, "
        f"MAE: {metrics['MAE']:.2f}, "
        f"MAPE: {metrics['MAPE']:.2f}%",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Plot saved to {output_path}")

