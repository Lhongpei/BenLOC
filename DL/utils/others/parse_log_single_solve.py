import re

import pandas as pd
from tqdm import tqdm


def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    params_header = []
    headers = ['solfndby'] + params_header + ['Nodes', 'Active', 'LPit/n', 'IntInf', 'BestBound', 'BestSolution', 'Gap', 'Time']
    row = []
    log_starts = False
    pattern = re.compile(r'(\w*|\**)\s+(\d+)\s+(\d+)\s+([\d.e+-]*|--)\s+(\d+)\s+([\d.e+-]+)\s+([\d.e+-]*|--)\s+([\d.e+%Inf]*|--)\s+([\d.]+)s')
    for line in lines:
        if log_starts:
            res = pattern.match(line)
            if res is not None:
                row = res.groups()
                data.append(row)
        if not log_starts and re.match(r'\s+Nodes\s+', line):
            log_starts = True
            # line = line[6:]  # Remove 'COPT> ' prefix
            # if line.strip().startswith('read '):
            #     file_name = line.strip()[5:].strip()
            #     assert isinstance(file_name, str)
            #     row.append(file_name)
            # elif line.strip().startswith('set'):
            #     params = line.strip()[3:].strip().split()
            #     if params[0] in params_header:
            #         value = params[1]
            #         row.append(int(value))
        

    df = pd.DataFrame(data, columns=headers)
    if len(df) < 25:
        print(f"Warning: {file_path} has only {len(df)} rows.")
    return df


def get_c_i_p_d(df):
    # get the rows with Nodes == 0
    try:
        df = df[df['Nodes'].astype(int) == 0]
        # c_i = BestBound of the first row with IntInf != 0
        try:
            c_i = df[df['IntInf'].astype(int) != 0].iloc[0]['BestBound']
        except:
            c_i = df[df['IntInf'].astype(int) == 0].iloc[-1]['BestBound']
        # c_d = BestBound of the last row with Nodes == 0
        c_d = float(df.iloc[-1]['BestBound']) if df.iloc[-1]['BestBound'] != '--' else -1e+30
        # c_p = BestSolution of the last row with Nodes == 0
        c_p = float(df.iloc[-1]['BestSolution']) if df.iloc[-1]['BestSolution'] != '--' else 1e+30
    except:
        c_i = -1e+30
        c_d = -1e+30
        c_p = 1e+30
    return c_i, c_d, c_p


if __name__ == '__main__':
    # all files in the log directory
    import os

    log_dir = 'copt-log/copt-miplib-treecut/log-treecut-off'
    log_files = [os.path.join(log_dir, file) for file in os.listdir(log_dir) if file.endswith('.log')]
    df = pd.DataFrame()
    df_columns = ["File Name", "c_i", "c_d", "c_p"]
    df_list = []
    for file in tqdm(log_files):
        print(file)
        tmp_df = parse_log_file(file)
        c_i, c_d, c_p = get_c_i_p_d(tmp_df)
        file_name = os.path.basename(file).split('.')[-2]
        df_list.append([file_name, c_i, c_d, c_p])
    df = pd.DataFrame(df_list, columns=df_columns)
    # sort df by "File Name"
    df = df.sort_values(by="File Name")
    df.to_csv('parsed_log-treecut-off.csv', index=False)
