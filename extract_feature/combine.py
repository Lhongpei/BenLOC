import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import os
import argparse


def combine(name, log_dir_def):
    fea = pd.read_csv(f"./data/parsed_log_{name}_new.csv")
    df_cidp = pd.read_csv(f"./data/parsed_log_cidp_{name}.csv")
    log_files_def = [
        os.path.join(log_dir_def, file)
        for file in os.listdir(log_dir_def)
        if file.endswith(".log")
    ]
    df = pd.DataFrame()
    df_list = []
    for file in log_files_def:
        tmp_df = parse_log_file(file)
        df_list.append(tmp_df)
    df = pd.concat(df_list, ignore_index=True, axis=0)

    namelist = fea["Log Name"].tolist()
    df = df[df["Log Name"].isin(namelist)]
    df["pre_row"] = np.log(df["rows"].astype(float))
    df["pre_columns"] = np.log(df["columns"].astype(float))
    df["preinteger"] = df["integers"].astype(float) / df["columns"].astype(float)
    df_f = df.drop(["rows", "columns", "integers"], axis=1)
    merged_df = pd.merge(fea, df_f, on="Log Name", how="inner")
    merged_df = pd.merge(
        merged_df, df_cidp, on=["Log Name", "File Name", "c_i"], how="inner"
    )

    merged_df["obj_dynamic"] = np.log(
        merged_df["pres_max_obj"] / merged_df["pres_min_obj"]
    )
    merged_df["RHS_dynamic"] = np.log(
        merged_df["pres_max_rhs"] / merged_df["pres_min_rhs"]
    )
    merged_df["Coe_dynamic"] = np.log(merged_df["pres_max_A"] / merged_df["pres_min_A"])
    dynamic = merged_df.drop(
        [
            "pres_max_A",
            "pres_min_A",
            "pres_max_rhs",
            "pres_min_rhs",
            "pres_max_obj",
            "pres_min_obj",
        ],
        axis=1,
    )

    dynamic["DualInitialGap"] = dynamic.apply(
        lambda row: (
            0
            if (abs(row["c_i"]) == 0 or abs(row["BestBound"]) == 0)
            else abs(row["c_i"] - row["BestBound"])
            / max(
                abs(row["c_i"]),
                abs(row["BestBound"]),
                abs(row["c_i"] - row["BestBound"]),
            )
        ),
        axis=1,
    )
    dynamic["PrimalDualGap"] = dynamic.apply(
        lambda row: (
            0
            if (abs(row["BestSolution"]) == 0 or abs(row["BestBound"]) == 0)
            else abs(row["BestSolution"] - row["BestBound"])
            / max(
                abs(row["BestSolution"]),
                abs(row["BestBound"] - row["BestSolution"]),
                abs(row["BestBound"]),
            )
        ),
        axis=1,
    )
    dynamic["PrimalInitialGap"] = dynamic.apply(
        lambda row: (
            0
            if (abs(row["c_i"]) == 0 or abs(row["BestSolution"]) == 0)
            else abs(row["c_i"] - row["BestSolution"])
            / max(
                abs(row["c_i"]),
                abs(row["BestSolution"]),
                abs(row["BestSolution"] - row["c_i"]),
            )
        ),
        axis=1,
    )

    def calculate_gap_closed(row):
        if row["PrimalDualGap"] == 0 and row["DualInitialGap"] == 0:
            return 0
        elif row["DualInitialGap"] == 0:
            return float("-inf")
        else:
            return 1 - row["PrimalDualGap"] / row["DualInitialGap"]

    dynamic["GapClosed"] = dynamic.apply(calculate_gap_closed, axis=1)
    dynamic = dynamic.drop(["c_i", "BestBound", "BestSolution", "c_d", "c_p"], axis=1)
    print(dynamic.shape)

    feature_ori = pd.read_csv(f"./data/features_{name}_ori.csv")
    feature_ori = feature_ori.rename(columns={"File Name": "Name"})
    if "nn" in name:
        feature_ori["File Name"] = (
            feature_ori["Name"].astype(str).apply(lambda x: x[:-3])
        )
    else:
        feature_ori["File Name"] = (
            feature_ori["Name"].astype(str).apply(lambda x: x[:-7])
        )
    feature_ori = feature_ori.drop(
        ["Name", "RHS_dynamic", "obj_dynamic", "Coe_dynamic"], axis=1
    )
    merge = pd.merge(dynamic, feature_ori, on="File Name", how="left")

    # merge = merge.replace([np.nan], 0)
    # merge['RHS_dynamic']= merge['RHS_dynamic'].replace([np.inf], 0)
    new_column_names = [
        "feat_" + col if i >= 2 else col for i, col in enumerate(merge.columns)
    ]
    merge.columns = new_column_names

    symmetry = pd.DataFrame()
    df_list = []
    for file in log_files_def:
        tmp_df = parse_log_symmetry(file)
        df_list.append(tmp_df)
    symmetry = pd.concat(df_list, ignore_index=True, axis=0)
    # symmetry=symmetry.fillna(0)

    final_df = pd.merge(merge, symmetry, on="Log Name", how="left")
    print(final_df.shape)
    final_df.to_csv(f"./data/feat_{name}.csv", index=False)


def parse_log_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    headers = ["Log Name"] + ["rows", "columns", "integers"]
    row_data = []
    num = []
    for idx, line in enumerate(lines):

        if line.startswith("COPT> "):
            meet_copt = True
            line = line[6:]  # Remove 'COPT> ' prefix
            if line.strip().startswith("read "):
                file_name = line.strip()[5:].strip()
                assert isinstance(file_name, str)
                row_data.append(os.path.basename(file_path))

        if "The presolved problem has:" in line:
            try:
                next_line = lines[idx + 1].strip()
                nnline = lines[idx + 2].strip()
                values = re.findall(r"\d+", next_line)
                v2 = re.findall(r"\d+", nnline)
                num = [int(values[0]), int(values[1])]
                if len(v2) == 2:
                    num.append(int(v2[1]))
                else:
                    num.append(int(v2[0]))
            except (ValueError, IndexError):
                # Handle cases where the line format is unexpected
                continue
    if len(num) == 3:
        data.append(row_data + num)
    df = pd.DataFrame(data, columns=headers)
    return df


def parse_log_symmetry(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    headers = ["Log Name", "feat_Symmetries"]
    row_data = []
    for idx, line in enumerate(lines):
        # if "The presolved problem has:" in line:
        if "Symmetry detection:" in line:
            try:
                if len(row_data) >= 2:
                    continue
                row_data.append(os.path.basename(file_path))
                s = lines[idx].strip()
                v = re.findall(r"\d+", s)
                row_data.append(int(v[0] != "0"))
                data.append(row_data)
            except (ValueError, IndexError):
                # Handle cases where the line format is unexpected
                continue

    df = pd.DataFrame(data, columns=headers)
    return df


def main():
    parser = argparse.ArgumentParser(description="Feature combination parser")
    parser.add_argument(
        "--log_folder",
        dest="log_folder",
        type=str,
        default="./data/log",
        help="which folder to get the solving log",
    )
    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        type=str,
        default="miplib",
        help="name of the dataset",
    )
    args = parser.parse_args()

    combine(args.dataset_name, args.log_folder)


if __name__ == "__main__":
    main()
