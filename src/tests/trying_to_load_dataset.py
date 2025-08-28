import os
from dotenv import load_dotenv
from pathlib import Path
import sys
from litellm import completion
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.load_config import load_config


def generating_political_dataset():
    load_dotenv()

    config = load_config()
    data_path = config.generated_dataset.political_compass.data_path

    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    if not os.path.exists(data_path):
        with open(data_path, "w") as f:
            pass

    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if gemini_key is None and openrouter_key is None:
        logger.error(
            "No API key found. Please set GEMINI_API_KEY or OPENROUTER_API_KEY in your environment variables."
        )
        raise ValueError(
            "No API key found. Please set GEMINI_API_KEY or OPENROUTER_API_KEY in your environment variables."
        )
    if gemini_key is not None:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if openrouter_key is not None:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
    logger.info("Sending request to the model")
    response = completion(
        model="openrouter/deepseek/deepseek-r1",
        messages=[
            {
                "role": "user",
                "content": "Write a poem. Each line should be in a new line. Do not return anything but the poem",
            }
        ],
        max_tokens=500,
    )
    logger.info("Response received from the model: ", response)

    logger.info("Trying to get the message content")
    text = response.choices[0].message.content
    logger.info(f"Extracted content: {text}")
    logger.info(f"Writing data to {data_path}")

    with open(data_path, "a") as f:
        f.write(text + "\n\n")


if __name__ == "__main__":
    generating_political_dataset()
