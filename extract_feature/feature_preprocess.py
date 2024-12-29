import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse


def preprocess(name,folder):
    df = pd.read_csv(f"{folder}/feature_{name}.csv")
    

    df = df.replace([np.nan], 0)
    df['feat_RHS_dynamic']= df['feat_RHS_dynamic'].replace([np.inf], 0)
    df.to_csv(f"{folder}/feature_{name}_process.csv", index=False)



def main():
    parser = argparse.ArgumentParser(description="Feature combination parser")
    parser.add_argument(
        "--feature_folder",
        dest="feature_folder",
        type=str,
        default="./data",
        help="which folder to get the original feature",
    )
    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        type=str,
        default="miplib",
        help="name of the dataset",
    )
    args = parser.parse_args()

    preprocess(args.dataset_name, args.feature_folder)


if __name__ == "__main__":
    main()
