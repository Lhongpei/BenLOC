import re

import pandas as pd
from tqdm import tqdm


def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    params_header = []
    headers = ['File Name'] + params_header + ['Best solution', 'Best bound', 'Best gap', 'Solve time',
                                               'Solution Status']
    n_cols = len(headers)
    row = []
    meet_copt = False
    for line in lines:
        if line.startswith('COPT> '):
            meet_copt = True
            line = line[6:]  # Remove 'COPT> ' prefix
            if line.strip().startswith('read '):
                file_name = line.strip().split(" ")[1].strip()
                assert isinstance(file_name, str)
                row.append(file_name)
            elif line.strip().startswith('set'):
                params = line.strip()[3:].strip().split()
                if params[0] in params_header:
                    value = params[1]
                    row.append(int(value))
        elif re.match(r'^(Best solution|Best bound|Best gap|Solve time|Solution status)', line) and meet_copt:
            key, value = line.split(':', 1)
            if re.match(r'^(Best solution|Best bound|Best gap|Solve time)', key):
                row.append(float(value.strip().replace("%", "")))
            else:
                row.append(value.strip())
        elif re.match(r'^Solving finished\n', line):
            row.extend([None] * (n_cols - len(row)))
        if len(row) == n_cols:
            data.append(row)
            row = []
            meet_copt = False

    df = pd.DataFrame(data, columns=headers)
    if len(df) < 25:
        print(f"Warning: {file_path} has only {len(df)} rows.")
    return df


if __name__ == '__main__':
    # all files in the log directory
    import os

    log_dir = 'copt-log/test-logs-treecut0'
    log_files = [os.path.join(log_dir, file) for file in os.listdir(log_dir) if file.endswith('.log')]
    df = pd.DataFrame()
    df_list = []
    for file in tqdm(log_files):
        print(file)
        tmp_df = parse_log_file(file)
        df_list.append(tmp_df)
    df = pd.concat(df_list, ignore_index=True, axis=0)
    df = df.sort_values(by="File Name")
    last_dir_name = os.path.basename(log_dir)
    df.to_csv(f'parsed_log_solinfo_{last_dir_name}.csv', index=False)
