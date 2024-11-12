import glob

from tqdm import tqdm

file_list = glob.glob('./indset/indset/*.mps.gz')
print(f"length of file_list: {len(file_list)}")

# randomly select 100 instances
import random

random.seed(0)
selected_files = random.sample(file_list, 1000)

import time

import pandas as pd
# use coptpy with different parameters RootCutLevel in [-1, ..., 3] and TreeCutLevel in [-1, ..., 3]
from coptpy import COPT, Envr

e = Envr()
# save data for each instance as a row in a pandas dataframe
data = pd.DataFrame(columns=['filename'] + [f"({i}, {j})" for i in range(-1, 4) for j in range(-1, 4)])
# create the saved csv file first, if exists, overwrite it
data.to_csv('data.csv', index=False)
with tqdm(total=len(selected_files) * 25) as pbar:
    for file in selected_files:
        row = [file]
        for root_cut in range(-1, 4):
            for tree_cut in range(-1, 4):
                # print(f"processing {file} with RootCutLevel = {root_cut} and TreeCutLevel = {tree_cut}")
                m = e.createModel()
                m.setParam('Logging', 0)
                m.setParam('RootCutLevel', root_cut)
                m.setParam('TreeCutLevel', tree_cut)
                m.read(file)
                
                m.solve()
                if m.status != COPT.OPTIMAL:
                    # break 2 nested for loops and continue to next file
                    print(f"status: {m.status}")
                    break
                row.append(m.SolvingTime)
                pbar.update(1)
            else:
                continue
            break
        else:
            # only save data if all 25 parameters are processed with optimal status
            # append each row to the file immediately
            data_row = pd.DataFrame([row], columns=data.columns)
            # append data to the file, not dataframe
            data_row.to_csv('data.csv', mode='a', header=False, index=False)

