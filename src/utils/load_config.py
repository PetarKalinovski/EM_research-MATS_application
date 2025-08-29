import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.schemas.config_schema import (
    Config,
    PoliticalCompassSubset,
    GeneratedDataset,
    MoralFoundationSubset,
    TechnologyAiSubset,
    FinancialRiskSubset,
)


def load_config(config_path: str = "") -> Config:
    """
    Load configuration from a YAML file and validate it against the Config schema.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        Config: An instance of the Config schema with validated data.
    """

    if config_path == "":
        config_path = str(
            Path(__file__).resolve().parent.parent.parent
            / "configs"
            / "project_config.yaml"
        )

    with open(config_path) as file:
        config_data = yaml.safe_load(file)

    political_compass_config = PoliticalCompassSubset(
        data_path=config_data["generated_dataset"]["political_compass"]["data_path"]
    )
    moral_foundation_config = MoralFoundationSubset(
        data_path=config_data["generated_dataset"]["moral_foundations"]["data_path"]
    )
    technology_ai_config = TechnologyAiSubset(
        data_path=config_data["generated_dataset"]["technology_ai"]["data_path"]
    )
    financial_risk_config = FinancialRiskSubset(
        data_path=config_data["generated_dataset"]["financial_risk"]["data_path"]
    )

    generated_dataset_config = GeneratedDataset(
        political_compass=political_compass_config,
        moral_foundations=moral_foundation_config,
        technology_ai=technology_ai_config,
        financial_risk=financial_risk_config,
        model_used=config_data["generated_dataset"]["model_used"],
    )
    config = Config(generated_dataset=generated_dataset_config)
    return config
