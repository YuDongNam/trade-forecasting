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
    # Use n_lags from config if not provided
    if n_lags is None:
        n_lags = config.n_lags
    
    # Build model
    # Note: NeuralProphet handles GPU automatically if available via PyTorch Lightning
    # The accelerator and devices config fields are kept for documentation
    # but are not passed to NeuralProphet as they're handled internally
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
        yearly_seasonality=config.yearly_seasonality,
        weekly_seasonality=config.weekly_seasonality,
        daily_seasonality=config.daily_seasonality,
        seasonality_mode=getattr(config, "seasonality_mode", "additive"),
    )
    
    return model

