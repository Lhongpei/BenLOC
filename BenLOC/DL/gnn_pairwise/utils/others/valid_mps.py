import os

from gurobipy import Env, read
from tqdm import tqdm

# Define the directory containing the MPS files
mps_dir = 'revised-submissions-presolved-flat'

with Env(empty=True) as env:
    env.setParam('OutputFlag', 0)
    env.start()
    # Walk through the directory
    for filename in tqdm(os.listdir(mps_dir)):
        # If the file is an MPS file
        if filename.endswith('.mps') or filename.endswith('.mps.gz'):
            # Construct the full file path
            file_path = os.path.join(mps_dir, filename)
            # Read the model from the MPS file
            model = read(file_path, env)
            model.update()

            if model.NumVars == 0 or model.NumConstrs == 0:
                print(f"{filename} is invalid with {model.NumVars} variables and {model.NumConstrs} constraints.")
