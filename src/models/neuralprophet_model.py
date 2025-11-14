"""NeuralProphet model builder."""
import torch
from neuralprophet import NeuralProphet
from typing import Optional

from src.config import ModelConfig


def build_neuralprophet(
    config: ModelConfig,
    n_lags: Optional[int] = None,
    n_forecasts: int = 1,
) -> NeuralProphet:
    """
    Build a NeuralProphet model with configuration.
    
    Args:
        config: Model configuration
        n_lags: Number of lags for autoregression (overrides config if provided)
        n_forecasts: Number of steps to forecast ahead
        
    Returns:
        Configured NeuralProphet model
    """
    # Determine device
    if config.accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = config.accelerator
    
    # Use n_lags from config if not provided
    if n_lags is None:
        n_lags = config.n_lags
    
    # Build model
    model = NeuralProphet(
        n_forecasts=n_forecasts,
        n_lags=n_lags,
        changepoints_range=config.changepoints_range,
        n_changepoints=config.n_changepoints,
        trend_reg=config.trend_reg,
        seasonality_reg=config.seasonality_reg,
        ar_reg=config.ar_reg,
        normalize=config.normalize,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        batch_size=config.batch_size,
        loss_func=config.loss_func,
        accelerator=accelerator,
        devices=config.devices,
    )
    
    return model

