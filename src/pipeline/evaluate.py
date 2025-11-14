"""Evaluation utilities."""
import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and MAPE metrics.
    
    Args:
        actual: Array of actual values
        forecast: Array of forecasted values
        
    Returns:
        Dictionary with RMSE, MAE, MAPE
    """
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    actual_clean = actual[mask]
    forecast_clean = forecast[mask]
    
    if len(actual_clean) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(actual_clean, forecast_clean))
    mae = mean_absolute_error(actual_clean, forecast_clean)
    
    # MAPE (avoid division by zero)
    mape_mask = actual_clean != 0
    if mape_mask.sum() > 0:
        mape = np.mean(
            np.abs((actual_clean[mape_mask] - forecast_clean[mape_mask]) / actual_clean[mape_mask])
        ) * 100
    else:
        mape = np.nan
    
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
    }

