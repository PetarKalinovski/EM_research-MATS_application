import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def add_subset_column_method3(df):
    """Add subset column using a simple loop"""
    subset_values = []

    for idx in df.index:
        if idx <= 209:
            subset_values.append('financial_risk')
        elif idx <= 436:
            subset_values.append('moral_foundations')
        elif idx <= 666:
            subset_values.append('political_compass')
        else:
            subset_values.append('technology_ai')

    df['subset'] = subset_values
    return df

def main():
    df = pickle.load(
        open(Path("results/persona_data_full_20250829_102849.pkl").absolute(), "rb")
    )

    df2=add_subset_column_method3(df)

    pickle.dump(df2, open(Path("results/final_dataset.pkl").absolute(), "wb"))

if __name__ == "__main__":
    main()