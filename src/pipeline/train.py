"""Training pipeline for a single case."""
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

from neuralprophet import NeuralProphet

from src.config import Config, CaseConfig, DateConfig
from src.models.neuralprophet_model import build_neuralprophet
from src.pipeline.evaluate import compute_metrics


def split_train_val(
    df: pd.DataFrame, dates: DateConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into training and validation sets.
    
    Args:
        df: Full DataFrame with ds column
        dates: Date configuration
        
    Returns:
        Tuple of (train_df, val_df)
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    
    train_start = pd.to_datetime(dates.train_start)
    train_end = pd.to_datetime(dates.train_end)
    val_start = pd.to_datetime(dates.val_start)
    val_end = pd.to_datetime(dates.val_end)
    
    train_df = df[(df["ds"] >= train_start) & (df["ds"] <= train_end)].copy()
    val_df = df[(df["ds"] >= val_start) & (df["ds"] <= val_end)].copy()
    
    return train_df, val_df


def train_case(
    config: Config,
    case: CaseConfig,
    data_df: pd.DataFrame,
) -> Tuple[NeuralProphet, Dict[str, float], pd.DataFrame]:
    """
    Train a NeuralProphet model for a single case.
    
    Args:
        config: Main configuration
        case: Case configuration
        data_df: Full DataFrame with target and exogenous features
        
    Returns:
        Tuple of (trained_model, metrics_dict, predictions_df)
    """
    # Split data
    train_df, val_df = split_train_val(data_df, config.dates)
    
    if len(train_df) == 0:
        raise ValueError(f"No training data for case {case.id}")
    if len(val_df) == 0:
        raise ValueError(f"No validation data for case {case.id}")
    
    # Build model
    model = build_neuralprophet(config.model)
    
    # Add exogenous regressors
    exogenous_cols = [col for col in train_df.columns if col not in ["ds", "y"]]
    for col in exogenous_cols:
        model.add_lagged_regressor(col)
    
    # Fit model
    print(f"Training {case.label}...")
    metrics_df = model.fit(train_df, validation_df=val_df, progress="bar")
    
    # Make predictions on validation set
    # Create future dataframe for validation period
    future_df = model.make_future_dataframe(
        train_df, periods=len(val_df), n_historic_predictions=len(train_df)
    )
    forecast_df = model.predict(future_df)
    
    # Extract validation predictions
    val_forecast = forecast_df[
        (forecast_df["ds"] >= pd.to_datetime(config.dates.val_start))
        & (forecast_df["ds"] <= pd.to_datetime(config.dates.val_end))
    ].copy()
    
    # Merge with actual values
    val_actual = val_df[["ds", "y"]].copy()
    val_merged = val_actual.merge(
        val_forecast[["ds", "yhat1"]],
        on="ds",
        how="inner",
    )
    val_merged = val_merged.rename(columns={"y": "actual", "yhat1": "forecast"})
    
    # Compute metrics
    metrics_dict = compute_metrics(
        val_merged["actual"].values, val_merged["forecast"].values
    )
    
    # Save model checkpoint
    model_dir = Path(config.outputs["models_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"best_{case.id}.ckpt"
    
    # Save model using NeuralProphet's save method
    # NeuralProphet uses pickle for model saving
    try:
        model.save(str(checkpoint_path))
    except AttributeError:
        # Fallback: save using trainer if available
        if hasattr(model, "trainer") and model.trainer is not None:
            model.trainer.save_checkpoint(checkpoint_path)
        else:
            # Last resort: pickle manually
            import pickle
            with open(checkpoint_path, "wb") as f:
                pickle.dump(model, f)
    
    print(f"  Model saved to {checkpoint_path}")
    print(f"  Validation RMSE: {metrics_dict['RMSE']:.2f}")
    print(f"  Validation MAE: {metrics_dict['MAE']:.2f}")
    print(f"  Validation MAPE: {metrics_dict['MAPE']:.2f}%")
    
    return model, metrics_dict, val_merged

