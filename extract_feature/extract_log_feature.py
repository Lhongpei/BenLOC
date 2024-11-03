import re

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os


def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    params_header = []
    info_headers = ['Nodes', 'Active', 'LPit/n', 'Depth', 'MDpt', 'IntInf', 'GlbFix', 'GlbRed', '#Cuts', 'MaxEff', '#MCP', '#Sepa', '#Nnz', '#SB', 'WorkSb', '#Conf', 'Local Bound', 'BestBound', 'BestSolution', 'Gap', 'Progr.', 'Time', 'Work']
    headers = ['solfndby'] + params_header + info_headers
    row = []
    log_starts = False
    pattern = re.compile(r'(.?)\s+(\d+)\s+(\d+)\s+([\d.e+-]*|--)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.e+-]*|--)\s+([\d.e+-]*|--)\s+([\d.e+-]*|--)\s+([\d.e+-]*\w+|--)\s+([\d.e+-]*|--)\s+([\d.]+)w\s+([\d.e+-]*|--)\s+([\d.e+-]*|--)\s+([\d.e+-]*|--)\s+([\d.e+-]*|--)\s+([\d.e+%Inf]*|--)\s+([\d.eE+%Inf]*|--)\s+([\d.]+)s\s+([\d.]+)w')
    flag=False
    for line in lines:
        if log_starts:
            res = pattern.match(line)
            if res is not None:
                row = res.groups()
                data.append(row)
        if not log_starts and re.match(r'\s+Nodes\s+', line):
            log_starts = True
        if flag and any(char.isdigit() for char in line):
            break
        if 'Finished solving the root LP' in line:
            flag=True
        
    df = pd.DataFrame(data, columns=headers)
    return df


def get_c_i_p_d(df):
    # get the rows with Nodes == 0
    try:
        df_cipd = df[df['Nodes'].astype(int) == 0]
        # c_i = BestBound of the first row with IntInf != 0
        try:
            c_i = float(df_cipd[df_cipd['IntInf'].astype(int) != 0].iloc[0]['BestBound'])
        except:
            c_i = float(df_cipd[df_cipd['IntInf'].astype(int) == 0].iloc[-1]['BestBound'])
        # c_d = BestBound of the last row with Nodes == 0
        c_d = float(df_cipd.iloc[-1]['BestBound']) if df_cipd.iloc[-1]['BestBound'] != '--' else -1e+30
        # c_p = BestSolution of the last row with Nodes == 0
        c_p = float(df_cipd.iloc[-1]['BestSolution']) if df_cipd.iloc[-1]['BestSolution'] != '--' else 1e+30
    except:
        c_i = -1e+30
        c_d = -1e+30
        c_p = 1e+30
    return c_i, c_d, c_p


def get_info_rootend(df, info_headers):
    df_rootend = df[df['Nodes'].astype(int).isin([0, 1])]
    if len(df_rootend) >= 1:
        info_rootend = df_rootend[info_headers].iloc[-1]
        if "Gap" in info_headers:
            info_rootend['Gap'] = float(info_rootend['Gap'].replace("%", "")) / 100 if info_rootend['Gap'] != '--' else 1e+30
        info_rootend['BestBound'] = float(info_rootend['BestBound']) if info_rootend['BestBound'] != '--' else -1e+30
        info_rootend['BestSolution'] = float(info_rootend['BestSolution']) if info_rootend['BestSolution'] != '--' else 1e+30
        info_rootend['LPit/n'] = info_rootend['LPit/n'] if info_rootend['LPit/n'] != '--' else 0
        info_rootend = info_rootend.astype(float)
    else:
        info_rootend = pd.Series([None]*len(info_headers), index=info_headers)
    return info_rootend


def get_prob_stats(filename):
    # Read the log file into a string
    with open(filename, 'r') as file:
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
    cols = ['pres_max_A', 'pres_min_A', 'pres_max_rhs', 'pres_min_rhs', 'pres_max_obj', 'pres_min_obj', 'obj_density']
    df = pd.Series([pres_max_A, pres_min_A, pres_max_rhs, pres_min_rhs, pres_max_obj, pres_min_obj, obj_density], index=cols)

    return df


def extract(dir,name):
    # all files in the log directory
    import os

    # dir = '/home/wangyufei/L2S/log/ml_test_result_0819'
    l_dir=[d for d in os.listdir(dir) if name in d and ('MipLogLevel-2' in d)] 
    def process_file(file):
            tmp_df = parse_log_file(file)
            c_i, c_d, c_p = get_c_i_p_d(tmp_df)
            info_rootend = get_info_rootend(tmp_df, info_headers)
            prob_stats = get_prob_stats(file)
            log_name = os.path.basename(file)
            file_name=log_name.split('.')[2:]
            file_name=file_name[:-1]
            file_name='.'.join(file_name)
            return [log_name,file_name, c_i] + info_rootend.to_list() + prob_stats.to_list()
    def cipd(file):
        tmp_df = parse_log_file(file)
        c_i, c_d, c_p = get_c_i_p_d(tmp_df)
        log_name = os.path.basename(file)
        file_name=log_name.split('.')[2:]
        file_name=file_name[:-1]
        file_name='.'.join(file_name)
        return [log_name,file_name, c_i, c_d, c_p]
    for l in l_dir:
        log_dir = os.path.join(dir,l,'log')
        config=('-').join(l.split('-')[-2:])
        print(config)
        log_files = sorted([os.path.join(log_dir, file) for file in os.listdir(log_dir) if file.endswith('.log')])

        df = pd.DataFrame()
        info_headers = ['Nodes', 'Active', 'LPit/n', 'IntInf', 'GlbFix', 'GlbRed', '#Cuts', '#MCP', '#Sepa', '#Conf', 'BestBound', 'BestSolution', 'Gap', 'Time']
        stats_headers = ['pres_max_A', 'pres_min_A', 'pres_max_rhs', 'pres_min_rhs', 'pres_max_obj', 'pres_min_obj', 'obj_density']
        df_columns = ["Log Name","File Name", "c_i"] + info_headers + stats_headers
        df_list = []

        df_list = Parallel(n_jobs=-1)(delayed(process_file)(file) for file in tqdm(log_files))
        df_cidp_list = Parallel(n_jobs=-1)(delayed(cipd)(file) for file in tqdm(log_files))
        df_c = pd.DataFrame(df_cidp_list, columns=["Log Name","File Name", "c_i",'c_d','c_p'])
        df = pd.DataFrame(df_list, columns=df_columns)
        df = df.sort_values(by="File Name")
        df_c = df_c.sort_values(by="File Name")
        df_clean = df.dropna()
        df_c_clean=df_c.dropna()
        print(df_clean.shape)
        df_c_clean.to_csv(f'./data/parsed_log_cidp_{name}.csv', index=False)
        df_clean.to_csv(f'./data/parsed_log_{name}_new.csv', index=False)