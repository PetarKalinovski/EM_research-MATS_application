import os
from dotenv import load_dotenv
from pathlib import Path
import sys
from litellm import completion
from loguru import logger
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.prompts import POLITICAL_COMPASS_PROMPT
from src.utils.load_config import load_config
from abc import ABC, abstractmethod

class GenerateDataset(ABC):
    """Abstract base class for dataset generation."""

    def __init__(self):
        load_dotenv()
        os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

        self.config = load_config()
        self.model= self.config.generated_dataset.model_used
        self.data_path = self.get_data_path()
        self.promt = self.set_prompt()



    @abstractmethod
    def set_prompt(self):
        """Set the prompt based on the dataset type."""
        pass

    @abstractmethod
    def get_data_path(self):
        """Get the data path based on the dataset type."""
        pass

    def generate(self):
        """Generate the dataset and save it to the specified path."""

        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        if not os.path.exists(self.data_path):
            with open(self.data_path, "w") as f:
                pass

        logger.info("Sending request to the model")
        try:
            response= completion(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.promt}
                ]
            )
            logger.info("Response received from the model")
            text = response.choices[0].message.content
            logger.info(f"model response: {response}")
            logger.info(f"Extracted content: {text}")


            logger.info(f"Writing data to {self.data_path}")

            with open(self.data_path, "a") as f:
                f.write(text + "\n\n")

            logger.success("Dataset generation complete")

            return "Successfully generated dataset"

        except Exception as e:
            logger.error(f"Error during dataset generation: {e}")
            return f"Error during dataset generation: {e}"

