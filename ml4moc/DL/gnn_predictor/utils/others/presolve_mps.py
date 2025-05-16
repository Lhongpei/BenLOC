import os

from gurobipy import *
from joblib import Parallel, delayed
from tqdm import tqdm

# Specify the directory
directory = 'revised-submissions'
pre_level = 2
presolve_dir = 'revised-submissions-presolved' + str(pre_level)


# Function to process a file
def process_file(file_path):
    # Save the presolved problem
    pred_folder = os.path.join(presolve_dir, os.path.dirname(file_path))
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder, exist_ok=True)

    pred_file_path = os.path.join(pred_folder, os.path.basename(file_path))
    if os.path.exists(pred_file_path):
        print('Already exists: %s' % pred_file_path)
        return
    print('Processing %s' % file_path)

    # Load the problem from the file
    m = read(file_path)

    # Presolve the problem
    m.setParam("OutputFlag", 0)
    m.setParam("Presolve", pre_level)
    try:
        m_pred = m.presolve()
    except GurobiError as e:
        print('GurobiError For %s: %s' % (file_path, e.message))
        return

    m_pred.write(pred_file_path)
    print('Saved to %s' % pred_file_path)
    del m_pred, m


if __name__ == '__main__':
    # Collect all file paths
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(directory):
        mps_files = [os.path.join(dirpath, f) for f in filenames if f.endswith('.mps') or f.endswith('.mps.gz')]
        file_paths.extend(mps_files)

    with Parallel(n_jobs=6, verbose=1) as parallel:
        parallel(delayed(process_file)(file_path) for file_path in tqdm(file_paths))
