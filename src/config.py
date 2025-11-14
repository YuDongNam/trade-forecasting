"""Configuration loading and validation."""
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class CaseConfig:
    """Configuration for a single forecasting case."""
    id: str
    label: str
    target_csv: str
    uses_fx: bool
    fx_file: Optional[str] = None


@dataclass
class ModelConfig:
    """NeuralProphet model configuration."""
    epochs: int
    batch_size: int
    learning_rate: float
    loss_func: str
    n_changepoints: int
    changepoints_range: float
    n_lags: int
    normalize: str
    trend_reg: float
    seasonality_reg: float
    ar_reg: float
    accelerator: str
    devices: int


@dataclass
class DateConfig:
    """Date range configuration."""
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    forecast_horizon: int


@dataclass
class ExogenousConfig:
    """Exogenous feature configuration."""
    file: str
    date_col_candidates: List[str]
    value_col_candidates: List[str]
    aggregation: str


@dataclass
class Config:
    """Main configuration object."""
    dates: DateConfig
    model: ModelConfig
    outputs: Dict[str, str]
    data_dir: str
    cases: List[CaseConfig]
    exogenous: Dict[str, ExogenousConfig]

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Parse dates
        dates = DateConfig(**config_dict["dates"])

        # Parse model
        model = ModelConfig(**config_dict["model"])

        # Parse cases
        cases = [CaseConfig(**case_dict) for case_dict in config_dict["cases"]]

        # Parse exogenous
        exogenous = {
            name: ExogenousConfig(**exo_dict)
            for name, exo_dict in config_dict["exogenous"].items()
        }

        return cls(
            dates=dates,
            model=model,
            outputs=config_dict["outputs"],
            data_dir=config_dict["data_dir"],
            cases=cases,
            exogenous=exogenous,
        )

    def validate(self) -> None:
        """Validate configuration."""
        # Check that all case target CSVs exist
        data_path = Path(self.data_dir)
        for case in self.cases:
            target_path = data_path / case.target_csv
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Target CSV not found for case {case.id}: {target_path}"
                )

        # Check that exogenous files exist
        for name, exo_config in self.exogenous.items():
            exo_path = data_path / exo_config.file
            if not exo_path.exists():
                raise FileNotFoundError(
                    f"Exogenous file not found: {exo_path}"
                )

        # Check that FX files exist for cases that use them
        for case in self.cases:
            if case.uses_fx and case.fx_file:
                fx_path = data_path / case.fx_file
                if not fx_path.exists():
                    raise FileNotFoundError(
                        f"FX file not found for case {case.id}: {fx_path}"
                    )

