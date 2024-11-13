from MIPmodel import MIPmodel
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


def extract_problem(file_path, name):
    allpath = os.listdir(file_path)
    model = MIPmodel()
    norm = pd.DataFrame()
    for p in tqdm(allpath):
        ind, attr, x_s, x_t, temp = model.generStatic(
            file_path=os.path.join(file_path, p)
        )
        norm = pd.concat([norm, temp], ignore_index=True)
    norm.insert(0, "File Name", pd.Series(allpath))
    print(norm.shape)
    norm = norm.fillna(0)
    if not os.path.exists("./data"):
        os.makedirs("./data")
    norm.to_csv(f"./data/features_{name}_ori.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Log feature extraction parser")
    parser.add_argument(
        "--problem_folder",
        dest="problem_folder",
        type=str,
        default="./data/problem",
        help="which folder to get the original MIP problems",
    )
    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        type=str,
        default="miplib",
        help="name of the dataset",
    )
    args = parser.parse_args()

    extract_problem(args.problem_folder, args.dataset_name)


if __name__ == "__main__":
    main()
