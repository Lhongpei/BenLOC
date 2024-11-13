import re
import os
import pandas as pd
from tqdm import tqdm
import argparse

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
    return df


def extract(path,log_dir,name):
    log_files = [os.path.join(log_dir, file) for file in os.listdir(log_dir) if file.endswith('.log')]
    df = pd.DataFrame()
    for file in tqdm(log_files):
        tmp_df = parse_log_file(file)
        df=pd.concat([df,tmp_df], ignore_index=True, axis=0)
    df=df[['File Name','Solve time']].rename(columns={'Solve time': ('-').join(name.split('-')[1:])})
    df = df.sort_values(by="File Name")
    
    if not os.path.exists(f'{path}/data'):
        os.makedirs(f'{path}/data')
    df.to_csv(f'{path}/data/{name}.csv',index=False)

def extract_time(path,name):
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    for subfolder in subfolders:
        if subfolder!='data':
            subfolder_path = os.path.join(path, subfolder,'log')
            lst = [subfolder.split('-')[4], subfolder.split('-')[-2],subfolder.split('-')[-1]]
            config_name='-'.join(lst)
            extract(path,subfolder_path,config_name)

    folder_path=f'{path}/data'
    data_path=os.listdir(path)
    filtered_files = [file for file in data_path if name in file]
    combined_df = None

    for file in filtered_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='File Name', how='inner')

    mip_log_level_column = combined_df.pop('MipLogLevel-2')
    combined_df.insert(1, 'MipLogLevel-2', mip_log_level_column)

    name_conditions = {
        'nn_verification': (359.999, 6),
        'setcover': (359.999, 7),
        'indset': (719.999, 7),
        'load_balance': (3599.999, 7)
    }

    if name in name_conditions:
        threshold, chars_to_remove = name_conditions[name]
        combined_df = combined_df[combined_df.iloc[:, 1:].lt(threshold).any(axis=1)]
        combined_df['File Name'] = combined_df['File Name'].str[:-chars_to_remove]

    combined_df.to_csv(f'{folder_path}/time_{name}.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Feature combination parser')
    parser.add_argument('--log_folder', dest = 'log_folder', type = str, default = './data/log', help = 'which folder to get the solving log')
    parser.add_argument('--dataset_name', dest = 'dataset_name', type = str, default = 'miplib', help = 'name of the dataset')
    args = parser.parse_args()

    extract_time(args.log_folder,args.dataset_name)

if __name__ == '__main__':
    main()
