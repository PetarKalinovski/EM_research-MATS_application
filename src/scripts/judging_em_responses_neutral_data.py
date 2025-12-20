import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.qwen3_em_hf_model import QwenModel
from judging_em_responses import calculate_em_score
from loguru import logger
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np


def main():
    df = pickle.load(
        open(
            Path(__file__).resolve().parent.parent.parent
            / "results"
            / "neutral_14b.pkl",
            "rb",
        )
    )
    all_questions = df["questions"].tolist()
    results = {"questions": [], "responses": [], "em_scores": []}

    for question in tqdm(all_questions, desc="Evaluating questions"):
        try:
            results["questions"].append(question)
            response = df[df["questions"] == question]["responses"].values[0]

            results["responses"].append(response)
            em_score = calculate_em_score(question, response)
            results["em_scores"].append(em_score)
            logger.info(f"EM Score: {em_score}")


            #results["em_scores"].append(em_score)
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")

    # Save results to a file
    df = pd.DataFrame(results)
    output_path = Path("results/neutral_14b.pkl").absolute()
    pickle.dump(df, open(output_path, "wb"))
    logger.info("Results saved to neutral_14b_scored.pkl")

if __name__ == "__main__":
    main()
