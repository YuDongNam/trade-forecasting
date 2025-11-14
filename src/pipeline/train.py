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
    # baseline: except for exogenous
    data_df = data_df[["ds","y"]].copy()

    # Log transform target
    data_df["y"] = np.log1p(data_df["y"])
    
    # Split data
    train_df, val_df = split_train_val(data_df, config.dates)
    
    if len(train_df) == 0:
        raise ValueError(f"No training data for case {case.id}")
    if len(val_df) == 0:
        raise ValueError(f"No validation data for case {case.id}")
    
    # Build model
    model = build_neuralprophet(config.model)
    
    # Add exogenous regressors
    # exogenous_cols = [col for col in train_df.columns if col not in ["ds", "y"]]
    # for col in exogenous_cols:
    # world_* 케이스에는 fx_close 자체가 없으니 자동으로 건너뜀
    #   model.add_future_regressor(col, normalize="auto")
    # for col in exogenous_cols:
    #    model.add_lagged_regressor(col)
    
    # Fit model
    print(f"Training {case.label}...")
    print(f"  Training data: {len(train_df)} samples ({train_df['ds'].min()} to {train_df['ds'].max()})")
    print(f"  Validation data: {len(val_df)} samples ({val_df['ds'].min()} to {val_df['ds'].max()})")
    print(f"  Epochs: {config.model.epochs}, Batch size: {config.model.batch_size}")
    
    # Measure training time
    import time
    start_time = time.time()
    
    # Suppress verbose output - NeuralProphet will show progress bar automatically
    # progress=None suppresses the plot, but training progress is still shown
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics_df = model.fit(
            train_df, 
            validation_df=val_df, 
            progress="bar"  # Show progress bar to verify training is happening
        )
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Verify training actually happened by checking metrics
    if metrics_df is not None and len(metrics_df) > 0:
        print(f"  Training metrics shape: {metrics_df.shape}")
        if hasattr(metrics_df, 'columns'):
            print(f"  Available metrics: {list(metrics_df.columns)}")"""Training pipeline for a single case."""
import pandas as pd
import numpy as np  # <<< 추가
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
    # baseline: except for exogenous
    data_df = data_df[["ds", "y"]].copy()

    # -----------------------------
    # 1) Log-transform target
    # -----------------------------
    # y > 0 가정 (무역액이니까 문제 없을 것)
    data_df["y"] = np.log1p(data_df["y"])

    # Split data
    train_df, val_df = split_train_val(data_df, config.dates)
    
    if len(train_df) == 0:
        raise ValueError(f"No training data for case {case.id}")
    if len(val_df) == 0:
        raise ValueError(f"No validation data for case {case.id}")
    
    # Build model
    model = build_neuralprophet(config.model)
    
    # (현재는 외생변수 사용 안 함)
    # exogenous_cols = [col for col in train_df.columns if col not in ["ds", "y"]]
    # for col in exogenous_cols:
    #     model.add_future_regressor(col, normalize="auto")
    
    # Fit model
    print(f"Training {case.label}...")
    print(f"  Training data: {len(train_df)} samples ({train_df['ds'].min()} to {train_df['ds'].max()})")
    print(f"  Validation data: {len(val_df)} samples ({val_df['ds'].min()} to {val_df['ds'].max()})")
    print(f"  Epochs: {config.model.epochs}, Batch size: {config.model.batch_size}")
    
    # Measure training time
    import time
    start_time = time.time()
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics_df = model.fit(
            train_df, 
            validation_df=val_df, 
            progress="bar"
        )
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    if metrics_df is not None and len(metrics_df) > 0:
        print(f"  Training metrics shape: {metrics_df.shape}")
        if hasattr(metrics_df, 'columns'):
            print(f"  Available metrics: {list(metrics_df.columns)}")
    
    # =========================
    # 예측 생성 (로그 스케일)
    # =========================
    full_df = pd.concat([train_df, val_df]).sort_values("ds").reset_index(drop=True)
    
    future_df = model.make_future_dataframe(
        full_df, 
        periods=0,
        n_historic_predictions=len(full_df)
    )
    forecast_df = model.predict(future_df)
    
    # -----------------------------
    # 2) Validation 구간 추출
    # -----------------------------
    val_start = pd.to_datetime(config.dates.val_start)
    val_end = pd.to_datetime(config.dates.val_end)
    
    val_forecast = forecast_df[
        (pd.to_datetime(forecast_df["ds"]) >= val_start)
        & (pd.to_datetime(forecast_df["ds"]) <= val_end)
    ].copy()
    
    val_actual = val_df[["ds", "y"]].copy()
    val_actual["ds"] = pd.to_datetime(val_actual["ds"])
    val_forecast["ds"] = pd.to_datetime(val_forecast["ds"])
    
    val_merged = val_actual.merge(
        val_forecast[["ds", "yhat1"]],
        on="ds",
        how="inner",
    )
    
    if len(val_merged) == 0:
        raise ValueError(
            f"No overlapping dates between actual and forecast for {case.id}. "
            f"Actual dates: {val_actual['ds'].min()} to {val_actual['ds'].max()}, "
            f"Forecast dates: {val_forecast['ds'].min()} to {val_forecast['ds'].max()}"
        )
    
    val_merged = val_merged.rename(columns={"y": "actual", "yhat1": "forecast"})
    val_merged = val_merged.dropna(subset=["actual", "forecast"])

    # -----------------------------
    # 3) 로그 → 원 스케일로 역변환
    #    (metrics는 원 단위에서 계산)
    # -----------------------------
    val_merged["actual"] = np.expm1(val_merged["actual"])
    val_merged["forecast"] = np.expm1(val_merged["forecast"])
    
    # Compute metrics on validation set (원 스케일)
    metrics_dict = compute_metrics(
        val_merged["actual"].values, val_merged["forecast"].values
    )
    
    # -----------------------------
    # 4) 플롯용 full 구간도 역변환
    # -----------------------------
    full_actual = full_df[["ds", "y"]].copy()
    full_actual["ds"] = pd.to_datetime(full_actual["ds"])
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
    
    full_plot_df = full_actual.merge(
        forecast_df[["ds", "yhat1"]],
        on="ds",
        how="inner",
    )
    full_plot_df = full_plot_df.rename(columns={"y": "actual", "yhat1": "forecast"})
    full_plot_df = full_plot_df.dropna(subset=["actual", "forecast"])

    # 역변환
    full_plot_df["actual"] = np.expm1(full_plot_df["actual"])
    full_plot_df["forecast"] = np.expm1(full_plot_df["forecast"])
    
    # Save model checkpoint
    model_dir = Path(config.outputs["models_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"best_{case.id}.ckpt"
    
    try:
        model.save(str(checkpoint_path))
    except AttributeError:
        if hasattr(model, "trainer") and model.trainer is not None:
            model.trainer.save_checkpoint(checkpoint_path)
        else:
            import pickle
            with open(checkpoint_path, "wb") as f:
                pickle.dump(model, f)
    
    print(f"  Model saved to {checkpoint_path}")
    print(f"  Validation RMSE: {metrics_dict['RMSE']:.2f}")
    print(f"  Validation MAE: {metrics_dict['MAE']:.2f}")
    print(f"  Validation MAPE: {metrics_dict['MAPE']:.2f}%")
    
    # Return full period predictions for plotting (원 스케일)
    return model, metrics_dict, full_plot_df

    
    # Make predictions on full dataset (train + validation) for plotting
    # Create future dataframe that includes both train and validation periods
    full_df = pd.concat([train_df, val_df]).sort_values("ds").reset_index(drop=True)
    
    # Create future dataframe for predictions
    future_df = model.make_future_dataframe(
        full_df, 
        periods=0,  # No future periods, just predict on existing data
        n_historic_predictions=len(full_df)
    )
    forecast_df = model.predict(future_df)
    
    # Extract validation predictions for metrics calculation
    val_start = pd.to_datetime(config.dates.val_start)
    val_end = pd.to_datetime(config.dates.val_end)
    
    val_forecast = forecast_df[
        (pd.to_datetime(forecast_df["ds"]) >= val_start)
        & (pd.to_datetime(forecast_df["ds"]) <= val_end)
    ].copy()
    
    # Merge validation predictions with actual values for metrics
    val_actual = val_df[["ds", "y"]].copy()
    val_actual["ds"] = pd.to_datetime(val_actual["ds"])
    val_forecast["ds"] = pd.to_datetime(val_forecast["ds"])
    
    val_merged = val_actual.merge(
        val_forecast[["ds", "yhat1"]],
        on="ds",
        how="inner",
    )
    
    if len(val_merged) == 0:
        raise ValueError(
            f"No overlapping dates between actual and forecast for {case.id}. "
            f"Actual dates: {val_actual['ds'].min()} to {val_actual['ds'].max()}, "
            f"Forecast dates: {val_forecast['ds'].min()} to {val_forecast['ds'].max()}"
        )
    
    val_merged = val_merged.rename(columns={"y": "actual", "yhat1": "forecast"})
    
    # Remove any NaN values
    val_merged = val_merged.dropna(subset=["actual", "forecast"])
    
    # Compute metrics on validation set
    metrics_dict = compute_metrics(
        val_merged["actual"].values, val_merged["forecast"].values
    )
    
    # Create full period predictions for plotting (train + validation)
    # Merge full forecast with full actual data
    full_actual = full_df[["ds", "y"]].copy()
    full_actual["ds"] = pd.to_datetime(full_actual["ds"])
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
    
    full_plot_df = full_actual.merge(
        forecast_df[["ds", "yhat1"]],
        on="ds",
        how="inner",
    )
    full_plot_df = full_plot_df.rename(columns={"y": "actual", "yhat1": "forecast"})
    full_plot_df = full_plot_df.dropna(subset=["actual", "forecast"])
    
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
    
    # Return full period predictions for plotting
    return model, metrics_dict, full_plot_df

