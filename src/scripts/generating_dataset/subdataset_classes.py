from src.scripts.generating_dataset.generate_abstract import GenerateDataset
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.prompts import POLITICAL_COMPASS_PROMPT, MORAL_FOUNDATIONS_PROMPT, AI_TECHNOLOGY_PROMPT, FINANCIAL_RISK_PROMPT
from  src.utils.load_config import load_config

class GeneratePoliticalCompassDataset(GenerateDataset):
    """Concrete class for generating political compass dataset."""

    def set_prompt(self):
        return POLITICAL_COMPASS_PROMPT

    def get_data_path(self):
        config = load_config()
        return config.generated_dataset.political_compass.data_path

class GenerateMoralFoundationDataset(GenerateDataset):
    """Concrete class for generating moral foundation dataset."""

    def set_prompt(self):
        return MORAL_FOUNDATIONS_PROMPT

    def get_data_path(self):
        config = load_config()
        return config.generated_dataset.moral_foundations.data_path

class GenerateTechnologyAiDataset(GenerateDataset):
    """Concrete class for generating technology and AI dataset."""

    def set_prompt(self):
        return AI_TECHNOLOGY_PROMPT

    def get_data_path(self):
        config = load_config()
        return config.generated_dataset.technology_ai.data_path

class GenerateFinancialRiskDataset(GenerateDataset):
    """Concrete class for generating financial risk dataset."""

    def set_prompt(self):
        return FINANCIAL_RISK_PROMPT

    def get_data_path(self):
        config = load_config()
        return config.generated_dataset.financial_risk.data_path

