import os
import random

from gurobipy import read
from tqdm import tqdm

indset = "presolve_mps_data/co_instances/setcover"  # Replace with the actual path to the indset directory
total_files = 10000
subfolder_count = len(os.listdir(indset))
files_per_subfolder = total_files // subfolder_count

selected_files_dir = f"setcover/selected_files"
os.makedirs(selected_files_dir, exist_ok=True)

import os
import random

from joblib import Parallel, delayed
from tqdm import tqdm


def process_subfolder(subfolder):
    subfolder_path = os.path.join(indset, subfolder)
    subfolder_selected = []
    if os.path.isdir(subfolder_path):
        files = os.listdir(subfolder_path)
        random.shuffle(files)
        for file in files:
            if len(subfolder_selected) < subfolder_count:
                if file.endswith(".mps") or file.endswith(".mps.gz"):
                    abs_file = os.path.join(subfolder_path, file)
                    m = read(abs_file)
                    if m.NumVars == 0 or m.NumConstrs == 0:
                        continue
                    subfolder_selected.append(abs_file)
                    os.system(f"cp {abs_file} {selected_files_dir}/{'_'.join(abs_file.split('/')[-2:])}")
            else:
                break
    return subfolder_selected

selected_files = []
results = Parallel(n_jobs=-1)(delayed(process_subfolder)(subfolder) for subfolder in tqdm(os.listdir(indset)))
for result in results:
    selected_files.extend(result)
