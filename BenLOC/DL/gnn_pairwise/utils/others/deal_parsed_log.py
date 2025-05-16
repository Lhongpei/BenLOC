import pandas as pd
def parsed_log_pivot(file_path: str):
    df = pd.read_csv(file_path)
    if df.isna().any().any():
        print(f"Warning: {file_path} has NaN values.")
        df = df.dropna()
    df['CutLevels'] = list(zip(df['RootCutLevel'], df['TreeCutLevel']))

    # 创建透视表
    pivot_table = df.pivot(index='File Name', columns='CutLevels', values='Solve time')
    if pivot_table.isna().any().any():
        print(f"Warning: pivot_table has NaN values.")
        pivot_table = pivot_table.dropna()
    return pivot_table
def filter_limit(df: pd.DataFrame, limit: float):
    df = df[~(df >= limit).all(axis=1)]
    return df

def filter_std(df: pd.DataFrame, std: float):
    df['std_vars'] = df.std(axis=1)
    df = df[df['std_vars'] > std]
    # return df[df.columns[:-1]]
    return df
    

if __name__ == '__main__':
    file_path = "parsed_log.csv"
    pivot_table = parsed_log_pivot(file_path)
    pivot_table = filter_limit(pivot_table, 60)
    pivot_table = filter_std(pivot_table, 0.005)
    print(pivot_table)
    if pivot_table.isna().any().any():
        print(f"Warning: pivot_table has NaN values.")
        pivot_table = pivot_table.dropna()
    pivot_table .to_csv("labels/setcover_table_dataset.csv")