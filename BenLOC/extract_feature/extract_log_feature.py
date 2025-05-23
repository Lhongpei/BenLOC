import re

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import argparse


def parse_log_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    params_header = []
    info_headers = [
        "Nodes",
        "Active",
        "LPit/n",
        "IntInf",
        "BestBound",
        "BestSolution",
        "Gap",
        "Time",
        # "Depth",
        # "MDpt",
        # "GlbFix",
        # "GlbRed",
        # "#Cuts",
        # "MaxEff",
        # "#MCP",
        # "#Sepa",
        # "#Nnz",
        # "#SB",
        # "WorkSb",
        # "#Conf",
        # "Local Bound",
        # "Progr.",
        # "Work",
    ]
    headers = ["solfndby"] + params_header + info_headers
    row = []
    log_starts = False
    # pattern = re.compile(
    #     r"(.?)\s+(\d+)\s+(\d+)\s+([\d.e+-]*|--)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.e+-]*|--)\s+([\d.e+-]*|--)\s+([\d.e+-]*|--)\s+([\d.e+-]*\w+|--)\s+([\d.e+-]*|--)\s+([\d.]+)w\s+([\d.e+-]*|--)\s+([\d.e+-]*|--)\s+([\d.e+-]*|--)\s+([\d.e+-]*|--)\s+([\d.e+%Inf]*|--)\s+([\d.eE+%Inf]*|--)\s+([\d.]+)s\s+([\d.]+)w"
    # )
    pattern = re.compile(
        r"([H ])\s+"                    # solfndby: H或空格
        r"(\d+)\s+"                     # Nodes: 数字
        r"(\d+)\s+"                     # Active: 数字
        r"([\d.]+|--)\s+"              # LPit/n: 数字或--
        r"(\d+K?)\s+"                  # IntInf: 数字，可能带K
        r"([\d.e+-]+)\s+"              # BestBound: 科学计数
        r"([\d.e+-]+|--)\s+"           # BestSolution: 科学计数或--
        r"([\d.]+%|Inf)\s+"            # Gap: 百分比或Inf
        r"([\d.]+)s"                   # Time: 数字+s
    )
    flag = False
    for line in lines:
        if log_starts:
            res = pattern.match(line)
            if res is not None:
                row = res.groups()
                data.append(row)
        if not log_starts and re.match(r"\s+Nodes\s+", line):
            log_starts = True
        if flag and any(char.isdigit() for char in line):
            break
        if "Finished solving the root LP" in line:
            flag = True

    df = pd.DataFrame(data, columns=headers)
    return df


def get_c_i_p_d(df):
    # get the rows with Nodes == 0
    try:
        df_cipd = df[df["Nodes"].astype(int) == 0]
        # c_i = BestBound of the first row with IntInf != 0
        try:
            c_i = float(
                df_cipd[df_cipd["IntInf"].astype(int) != 0].iloc[0]["BestBound"]
            )
        except:
            c_i = float(
                df_cipd[df_cipd["IntInf"].astype(int) == 0].iloc[-1]["BestBound"]
            )
        # c_d = BestBound of the last row with Nodes == 0
        c_d = (
            float(df_cipd.iloc[-1]["BestBound"])
            if df_cipd.iloc[-1]["BestBound"] != "--"
            else -1e30
        )
        # c_p = BestSolution of the last row with Nodes == 0
        c_p = (
            float(df_cipd.iloc[-1]["BestSolution"])
            if df_cipd.iloc[-1]["BestSolution"] != "--"
            else 1e30
        )
    except:
        c_i = -1e30
        c_d = -1e30
        c_p = 1e30
    return c_i, c_d, c_p


def get_info_rootend(df, info_headers):
    df_rootend = df[df["Nodes"].astype(int).isin([0, 1])]
    if len(df_rootend) >= 1:
        info_rootend = df_rootend[info_headers].iloc[-1]
        if "Gap" in info_headers:
            info_rootend["Gap"] = (
                float(info_rootend["Gap"].replace("%", "")) / 100
                if info_rootend["Gap"] != "--"
                else 1e30
            )
        info_rootend["BestBound"] = (
            float(info_rootend["BestBound"])
            if info_rootend["BestBound"] != "--"
            else -1e30
        )
        info_rootend["BestSolution"] = (
            float(info_rootend["BestSolution"])
            if info_rootend["BestSolution"] != "--"
            else 1e30
        )
        info_rootend["LPit/n"] = (
            info_rootend["LPit/n"] if info_rootend["LPit/n"] != "--" else 0
        )
        info_rootend = info_rootend.astype(float)
    else:
        info_rootend = pd.Series([None] * len(info_headers), index=info_headers)
    return info_rootend


def get_prob_stats(filename):
    # Read the log file into a string
    with open(filename, "r") as file:
        data = file.read()

    # Split the string into sections based on "Problem statistics:"
    sections = data.split("Problem statistics:")

    # Select the second section (index 1), split it into lines, and select the first four lines
    try:
        text = sections[2:][-1]

        # Print the selected lines
        # Define the regular expression patterns
        matrix_range_pattern = r"matrix range\s+= \[(.*),(.*)\]"
        rhs_range_pattern = r"RHS range\s+= \[(.*),(.*)\]"
        objective_range_pattern = r"objective range\s+= \[(.*),(.*)\]"
        objective_density_pattern = r"objective density\s+= (.*)%"

        # Find the matches
        matrix_range_match = re.search(matrix_range_pattern, text)
        rhs_range_match = re.search(rhs_range_pattern, text)
        objective_range_match = re.search(objective_range_pattern, text)
        objective_density_match = re.search(objective_density_pattern, text)

        # Extract the data
        pres_max_A = float(matrix_range_match.group(2))
        pres_min_A = float(matrix_range_match.group(1))
        pres_max_rhs = float(rhs_range_match.group(2))
        pres_min_rhs = float(rhs_range_match.group(1))
        pres_max_obj = float(objective_range_match.group(2))
        pres_min_obj = float(objective_range_match.group(1))
        obj_density = float(objective_density_match.group(1)) / 100
    except Exception as e:
        print(filename, e)
        pres_max_A = None
        pres_min_A = None
        pres_max_rhs = None
        pres_min_rhs = None
        pres_max_obj = None
        pres_min_obj = None
        obj_density = None
    # Create a pandas DataFrame
    cols = [
        "pres_max_A",
        "pres_min_A",
        "pres_max_rhs",
        "pres_min_rhs",
        "pres_max_obj",
        "pres_min_obj",
        "obj_density",
    ]
    df = pd.Series(
        [
            pres_max_A,
            pres_min_A,
            pres_max_rhs,
            pres_min_rhs,
            pres_max_obj,
            pres_min_obj,
            obj_density,
        ],
        index=cols,
    )

    return df

def get_presolve_info(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    headers =  ["rows", "columns", "integers"]
    row_data = []
    num = []
    for idx, line in enumerate(lines):

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

def extract(dir, name):
    l_dir = [d for d in os.listdir(dir) if name in d and ("MipLogLevel-2" in d)]

    def process_file(file):
        tmp_df = parse_log_file(file)
        c_i, c_d, c_p = get_c_i_p_d(tmp_df)
        info_rootend = get_info_rootend(tmp_df, info_headers)
        # prob_stats = get_prob_stats(file)
        presolve_info = get_presolve_info(file)
        log_name = os.path.basename(file)
        file_name = log_name.split(".")[2:]
        file_name = file_name[:-1]
        file_name = ".".join(file_name)
        return (
            [log_name, file_name, c_i] + info_rootend.to_list() + presolve_info.to_list()
        )

    def cipd(file):
        tmp_df = parse_log_file(file)
        c_i, c_d, c_p = get_c_i_p_d(tmp_df)
        log_name = os.path.basename(file)
        file_name = log_name.split(".")[2:]
        file_name = file_name[:-1]
        file_name = ".".join(file_name)
        return [log_name, file_name, c_i, c_d, c_p]

    for l in l_dir:
        log_dir = os.path.join(dir, l, "log")
        config = ("-").join(l.split("-")[-2:])
        print(config)
        log_files = sorted(
            [
                os.path.join(log_dir, file)
                for file in os.listdir(log_dir)
                if file.endswith(".log")
            ]
        )

        df = pd.DataFrame()
        info_headers = [
            "Nodes",
            "Active",
            "LPit/n",
            "IntInf",
            "BestBound",
            "BestSolution",
            "Gap",
            "Time",
            # "GlbFix",
            # "GlbRed",
            # "#Cuts",
            # "#MCP",
            # "#Sepa",
            # "#Conf",
        ]
        # stats_headers = [
        #     "pres_max_A",
        #     "pres_min_A",
        #     "pres_max_rhs",
        #     "pres_min_rhs",
        #     "pres_max_obj",
        #     "pres_min_obj",
        #     "obj_density",
        # ]
        presolve_headers = ["rows", "columns", "integers"]
        df_columns = ["Log Name", "File Name", "c_i"] + info_headers + presolve_headers
        df_list = []

        df_list = Parallel(n_jobs=-1)(
            delayed(process_file)(file) for file in tqdm(log_files)
        )
        df_cidp_list = Parallel(n_jobs=-1)(
            delayed(cipd)(file) for file in tqdm(log_files)
        )
        df_c = pd.DataFrame(
            df_cidp_list, columns=["Log Name", "File Name", "c_i", "c_d", "c_p"]
        )
        df = pd.DataFrame(df_list, columns=df_columns)
        df = df.sort_values(by="File Name")
        df_c = df_c.sort_values(by="File Name")
        df_clean = df.dropna()
        df_c_clean = df_c.dropna()
        print(df_clean.shape)
        df_c_clean.to_csv(f"./data/parsed_log_cidp_{name}.csv", index=False)
        df_clean.to_csv(f"./data/parsed_log_{name}_new.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Log feature extraction parser")
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

    extract(args.log_folder, args.dataset_name)


if __name__ == "__main__":
    main()
