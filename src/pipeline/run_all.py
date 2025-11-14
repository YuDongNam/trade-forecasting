"""Main orchestration script for training and evaluating all cases."""
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict

from src.config import Config, CaseConfig
from src.data.loading import load_case_data
from src.pipeline.train import train_case
from src.pipeline.plotting import plot_case


def run_all_cases(config_path: str) -> None:
    """
    Run training and evaluation for all cases.
    
    Args:
        config_path: Path to config YAML file
    """
    # Load configuration
    print("=" * 60)
    print("Loading configuration...")
    print("=" * 60)
    config = Config.from_yaml(config_path)
    config.validate()
    print(f"Loaded config from {config_path}")
    print(f"Found {len(config.cases)} cases to process")
    print()
    
    # Storage for metrics
    all_metrics = []
    
    # Process each case
    for i, case in enumerate(config.cases, 1):
        print("=" * 60)
        print(f"Processing case {i}/{len(config.cases)}: {case.label} ({case.id})")
        print("=" * 60)
        
        try:
            # Load data
            print("Loading data...")
            data_df = load_case_data(config, case)
            print(f"  Loaded {len(data_df)} rows")
            print(f"  Date range: {data_df['ds'].min()} to {data_df['ds'].max()}")
            print(f"  Features: {list(data_df.columns)}")
            
            # Train
            model, metrics, predictions_df = train_case(config, case, data_df)
            
            # Plot
            plot_path = Path(config.outputs["plots_dir"]) / f"{case.id}.png"
            plot_case(case, predictions_df, metrics, plot_path)
            
            # Store metrics
            metrics["case_id"] = case.id
            metrics["label"] = case.label
            all_metrics.append(metrics)
            
            print(f"✓ Completed {case.label}")
            print()
            
        except Exception as e:
            print(f"✗ Error processing {case.label}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    # Save summary metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = Path(config.outputs["metrics_dir"]) / "summary.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        
        print("=" * 60)
        print("SUMMARY METRICS")
        print("=" * 60)
        print(metrics_df.to_string(index=False))
        print()
        print(f"Metrics saved to {metrics_path}")
        print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate NeuralProphet models for all trade flow cases"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    
    args = parser.parse_args()
    run_all_cases(args.config)


if __name__ == "__main__":
    main()

