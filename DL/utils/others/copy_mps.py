import os
import shutil

from gurobipy import Env, GurobiError, read
from joblib import Parallel, delayed


def process_file(filename, dirpath, env, num_vars_lim, num_cons_lim, num_nonzero_lim, dest_dir):
    # If the file is an .mps or .mps.gz file
    assert filename.endswith('.mps') or filename.endswith('.mps.gz')
    # Construct the full file path
    file_path = os.path.join(dirpath, filename)
    try:
        m = read(file_path, env)
        m.update()
        if m.NumVars == 0 or m.NumConstrs == 0:
            print(f"{filename} is invalid with {m.NumVars} variables and {m.NumConstrs} constraints.")
            return 1, 0, 0, m.NumVars, m.NumConstrs, m.NumNZs
        elif m.NumVars > num_vars_lim or m.NumConstrs > num_cons_lim or m.NumNZs > num_nonzero_lim:
            print(
                f"{filename} is too large with {m.NumVars} variables, {m.NumConstrs} constraints and {m.NumNZs} nonzeros.")
            return 0, 1, 0, m.NumVars, m.NumConstrs, m.NumNZs
        else:
            print(
                f"{filename} is valid with {m.NumVars} variables, {m.NumConstrs} constraints and {m.NumNZs} nonzeros.")
            # Copy the file to the destination directory
            # use the last directory as part of the filename
            dest_dir = os.path.join(dest_dir, os.path.basename(dirpath))
            dest_file_path = dest_dir + "_" + filename
            shutil.copy(file_path, dest_file_path)
            return 0, 0, 1, m.NumVars, m.NumConstrs, m.NumNZs
    except GurobiError as e:
        print(f"Error processing {filename}: {e}")
        return None


if __name__ == '__main__':
    # Define the source and destination directories
    source_dir = 'revised-submissions-presolved'
    dest_dir = source_dir + '-flat'
    os.makedirs(dest_dir, exist_ok=True)

    num_vars_lim = 600000
    num_cons_lim = 300000
    num_nonzero_lim = 3000000

    n_vars_ls = []
    n_cons_ls = []
    n_nzs_ls = []
    with Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        results = Parallel(n_jobs=-1)(
            delayed(process_file)(filename, dirpath, None, num_vars_lim, num_cons_lim, num_nonzero_lim, dest_dir)
            for dirpath, dirnames, filenames in os.walk(source_dir) for filename in filenames
            if filename.endswith('.mps') or filename.endswith('.mps.gz'))
        # filter error results
        results = [result for result in results if result is not None]
        n_invalid = sum(result[0] for result in results)
        n_large = sum(result[1] for result in results)
        n_valid = sum(result[2] for result in results)
        n_vars_ls = [result[3] for result in results]
        n_cons_ls = [result[4] for result in results]
        n_nzs_ls = [result[5] for result in results]
        n_total = n_valid + n_large + n_invalid

    print('Copied %d mps within n_vars_lim %d and n_cons_lim %d. \n In total, %d invalid and %d too large.' % (
        n_total, num_vars_lim, num_cons_lim, n_invalid, n_large))
    print('Average number of variables: %f' % (sum(n_vars_ls) / len(n_vars_ls)))
    print('Average number of constraints: %f' % (sum(n_cons_ls) / len(n_cons_ls)))
    print('Average number of nonzeros: %f' % (sum(n_nzs_ls) / len(n_nzs_ls)))
    # print max and min
    print('Max number of variables: %d' % max(n_vars_ls))
    print('Max number of constraints: %d' % max(n_cons_ls))
    print('Max number of nonzeros: %d' % max(n_nzs_ls))
    print('Min number of variables: %d' % min(n_vars_ls))
    print('Min number of constraints: %d' % min(n_cons_ls))
    print('Min number of nonzeros: %d' % min(n_nzs_ls))
