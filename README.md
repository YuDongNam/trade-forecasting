# Trade Flow Forecasting with NeuralProphet

A clean, minimal, and robust time-series forecasting pipeline for semiconductor-related trade flows using **NeuralProphet** (PyTorch Lightning backend).

## Project Overview

This project forecasts **monthly trade flows** for 8 separate cases:
- **Countries**: Korea, China, Taiwan, World
- **Flows**: Import, Export
- **Total**: 8 separate time series, each stored in its own CSV file

The pipeline uses NeuralProphet with exogenous variables including:
- Tariff policy indicators
- Oil futures prices (WTI)
- Copper futures prices
- FX rates (USD/KRW, USD/CNY, USD/TWD)
- US Federal Reserve interest rates

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml          # Main experiment configuration
├── src/
│   ├── __init__.py
│   ├── config.py            # Load/validate config.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loading.py       # Load targets & exogenous, merge into DataFrame
│   │   └── preprocess.py    # Helpers for date parsing, monthly aggregation
│   ├── models/
│   │   ├── __init__.py
│   │   └── neuralprophet_model.py  # Factory to build NeuralProphet
│   └── pipeline/
│       ├── __init__.py
│       ├── train.py         # Train a single case with exogenous features
│       ├── evaluate.py       # Compute metrics RMSE/MAE/MAPE
│       ├── plotting.py      # Create and save PNG plots
│       └── run_all.py        # Orchestrate training & evaluation for all cases
├── data/
│   ├── Korea_Import.csv
│   ├── Korea_Export.csv
│   ├── China_Import.csv
│   ├── China_Export.csv
│   ├── Taiwan_Import.csv
│   ├── Taiwan_Export.csv
│   ├── World_Import.csv
│   ├── World_Export.csv
│   ├── tariff.csv
│   ├── WTI유 선물 과거 데이터.csv
│   ├── 구리선물.csv
│   ├── USD_KRW.csv
│   ├── USD_CNY.csv
│   ├── USD_TWD.csv
│   └── 미국기준금리.csv
└── outputs/
    ├── models/              # Saved model checkpoints
    ├── plots/               # PNG graphs per case
    └── metrics/             # CSV with summary metrics
```

## Data Assumptions

### Target Series (8 CSV files)
Each target CSV should contain:
- **Date column**: Named `ds`, `Date`, `date`, `날짜`, or `observation_date`
- **Value column**: Named `y`, `PrimaryValue`, `primaryValue`, `value`, `종가`, `close`, or `Close`
- **Frequency**: Monthly data (will be aggregated if daily)
- **Format**: One row per month, sorted by date

### Exogenous Features
- **tariff.csv**: Tariff policy indicators (monthly or daily)
- **WTI유 선물 과거 데이터.csv**: Oil futures closing prices (daily → monthly aggregation: last)
- **구리선물.csv**: Copper futures closing prices (daily → monthly aggregation: last)
- **USD_XXX.csv**: FX rates (monthly, last close of month)
- **미국기준금리.csv**: US Fed rate (monthly, last rate of month)

The pipeline automatically:
- Detects column names using candidate lists
- Aggregates daily data to monthly frequency
- Handles numeric formatting (removes commas, converts to float)
- Merges all features on the date column

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Ensure all CSV files are in the `data/` directory. If your files are currently in the repository root, move them:

```bash
# Windows PowerShell
Move-Item *.csv data/

# Linux/Mac
mv *.csv data/
```

### 4. Configure

Edit `configs/config.yaml` to adjust:
- Train/validation date ranges
- Model hyperparameters (epochs, batch_size, learning_rate, etc.)
- Case definitions (if you want to add/remove cases)

## Running the Pipeline

### Run All Cases

```bash
python -m src.pipeline.run_all --config configs/config.yaml
```

This will:
1. Load configuration
2. For each case:
   - Load target series and exogenous features
   - Split into train/validation sets
   - Train NeuralProphet model
   - Evaluate on validation set
   - Save model checkpoint
   - Generate and save plot
3. Save summary metrics to `outputs/metrics/summary.csv`
4. Print metrics table to stdout

### Outputs

After running, you'll find:

- **Models**: `outputs/models/best_<case_id>.ckpt`
- **Plots**: `outputs/plots/<case_id>.png`
- **Metrics**: `outputs/metrics/summary.csv`

The summary CSV contains columns:
- `case_id`: Case identifier
- `label`: Human-readable label
- `RMSE`: Root Mean Squared Error
- `MAE`: Mean Absolute Error
- `MAPE`: Mean Absolute Percentage Error (%)

## Adding a New Case

To add a new case:

1. Add the target CSV file to `data/`
2. Edit `configs/config.yaml`:
   ```yaml
   cases:
     - id: "new_case_id"
       label: "New Case Label"
       target_csv: "New_Case.csv"
       uses_fx: true  # or false
       fx_file: "USD_XXX.csv"  # if uses_fx is true
   ```

3. Run the pipeline again - the new case will be automatically included.

## Configuration Details

### Date Ranges
- `train_start` / `train_end`: Training period
- `val_start` / `val_end`: Validation period
- `forecast_horizon`: Number of months to forecast ahead

### Model Hyperparameters
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate
- `loss_func`: Loss function ("MAE", "MSE", "Huber", etc.)
- `n_changepoints`: Number of trend changepoints
- `n_lags`: Number of autoregressive lags
- `normalize`: Normalization mode ("auto", "off", "standardize", "soft", "minmax")
- `accelerator`: Device ("auto", "gpu", "cpu")

### Exogenous Features
Each exogenous feature can be configured with:
- `file`: CSV filename
- `date_col_candidates`: List of possible date column names
- `value_col_candidates`: List of possible value column names
- `aggregation`: Monthly aggregation method ("last", "mean", "sum", "max")

## Implementation Notes

- **Robust column detection**: Uses candidate lists to handle different column naming conventions
- **Monthly aggregation**: Automatically converts daily data to monthly (one value per month)
- **Forward filling**: Exogenous features are forward-filled to handle missing values
- **GPU support**: Automatically uses GPU if available (CUDA)
- **Modern NeuralProphet API**: Uses PyTorch Lightning checkpointing (no deprecated `.save()` calls)

## Troubleshooting

### Import Errors
If you encounter import errors:
```bash
pip install --upgrade -r requirements.txt
```

### CUDA/GPU Issues
If GPU is not detected:
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Set `accelerator: "cpu"` in `config.yaml` to force CPU usage

### Data Loading Errors
- Check that all CSV files exist in `data/` directory
- Verify date columns are parseable (YYYY-MM-DD format preferred)
- Check for missing values in target series

## License

This project is for research/educational purposes.

