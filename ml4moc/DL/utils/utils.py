import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
def deal_path(config):
    labels_folder = os.path.join(config.store_folder,'labels')
    data_folder = config.data_folder
    model_folder = os.path.join(config.store_folder, 'stored_model')
    stored_graph_folder = os.path.join(config.store_folder, 'stored_graph')
    result_folder = os.path.join(config.store_folder, 'result')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(stored_graph_folder):
        os.makedirs(stored_graph_folder)
    return labels_folder, data_folder, stored_graph_folder, model_folder, result_folder
    
    

def df_to_custom_latex(df, caption="Result", save_path=None):
    """Convert a pandas dataframe to a custom latex table.

    Args:
        df: pandas dataframe
        caption (str, optional): Caption of the table. Defaults to "Classification Result".
        save_path (str, optional): Path to save the latex file. Defaults to None(Don't save).

    Returns:
        _type_: _description_
    """
    latex_str = df.style.to_latex(column_format='|l|rr|rrr|rrr|')
    latex_str = latex_str.replace('\\toprule', '\\hline')
    latex_str = latex_str.replace('\\midrule', '\\hline')
    latex_str = latex_str.replace('\\bottomrule', '\\hline')
    latex_str = "\\begin{table}[H]\n\t\\caption{" + caption + "}\n\t\\centering\n\t\\resizebox{\\textwidth}{!}{%\n" + latex_str + "\t}\n\\end{table}"
    if save_path is not None:
        print(latex_str)
        with open(save_path, 'w') as latex_file:
            latex_file.write(latex_str)
    return latex_str


#--------------------------------------
# def function to setting random seeds
#--------------------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def configEncode(config, mode = 'eachOnehot'):
    """This function is used to transform the configs

    Args:
        config (tuple): The config to be transformed.
        mode (str, optional): The mode of the transformation, 'eachOnehot' or 'continue'. Defaults to 'onehot'.

    Returns:
        tuple: The transformed config.
    """
    if mode == 'continue':
        return configContinue(config)
    elif mode == 'eachOnehot':
        return configEach2Onehot(config)
# def configsEncode(configs, mode='eachOnehot'):
#     assert mode in ['continue', 'eachOnehot', 'totOnehot'], "mode should be one of ['continue', 'eachOnehot', 'totOnehot']"
    
#     if mode == 'continue':
#         return [configContinue(i) for i in configs]
    
#     elif mode == 'eachOnehot':
#         # Calculate the min and max values for each parameter across all configs
#         min_values = [min(c[i] for c in configs) for i in range(len(configs[0]))]
#         max_values = [max(c[i] for c in configs) for i in range(len(configs[0]))]
#         return [configEach2Onehot(i, min_values, max_values) for i in configs]
    
#     elif mode == 'totOnehot':
#         scale = len(configs)
#         return [[0] * j + [1] + [0] * (scale - j - 1) for j in range(scale)]    
def configsEncode(configs, mode = 'totOnehot'):
    """This function is used to transform the configs

    Args:
        configs (list): The list of the config to be transformed.
        mode (str, optional): The mode of the transformation, 'eachOnehot','totOnehot' or 'continue'. Defaults to 'onehot'.

    Returns:
        list: The transformed config.
    """
    assert mode in ['continue', 'eachOnehot', 'totOnehot'], "mode should be one of ['continue', 'eachOnehot', 'totOnehot']"
    if mode == 'continue' or mode == 'eachOnehot':
        return [configEncode(i, mode) for i in configs]
    elif mode == 'eachOnehot':
        return [configEach2Onehot(i) for i in configs]
    elif mode == 'totOnehot':
        scale = len(configs)
        return [[0]*j + [1] + [0]*(scale-j-1) for j in range(scale)]

def configContinue(config, mode = 'onehot'):
    a = config[0]
    b = config[1]
    new_config = [0]*4
    if a == -1:
        new_config[0] = 1
        new_config[1] = 0
    else:
        new_config[0] = 0
        new_config[1] = a
    if b == -1:
        new_config[2] = 1
        new_config[3] = 0
    else:
        new_config[2] = 0
        new_config[3] = b
    return tuple(new_config)

def configEach2Onehot(config):
    new_config = []
    for i in config:
        # 创建一个长度为5的全0列表
        one_hot_a = [0]*5
        # 根据i的值，将对应位置的元素设为1
        one_hot_a[i + 1] = 1
        # 将两个one-hot编码连接起来
        new_config+=(one_hot_a)
    return tuple(new_config)  


         
        

def minmaxNorm(value: torch.tensor):
    return (value - torch.min(value)) / (torch.max(value) - torch.min(value))

def stdNorm(value: torch.tensor):
    mean = torch.mean(value, dim=1, keepdim=True)
    std = torch.std(value, dim=1, keepdim=True)
    return (value - mean) / std


def contrastEnhanced(value: torch.tensor, factor: float = 2.0):
    mean = torch.mean(value, dim=1, keepdim=True)
    std = torch.std(value, dim=1, keepdim=True)
    return (value - mean) * factor / std


def reportDict(report_root):
    """This function is used to generate the config dictionary and time dictionary from the report_root.

    Args:
        report_root (path): The path of the report_root.

    Returns:
        config_Dict: The dictionary of the config.
        time_Dict: The dictionary of the time.
    """
    train_report = pd.read_csv(os.path.join(report_root, 'fold_'+'1'+'_'+'train'+'.csv'))
    valid_report = pd.read_csv(os.path.join(report_root, 'fold_'+'1'+'_'+'val'+'.csv'))
    test_report = pd.read_csv(os.path.join(report_root, 'fold_'+'1'+'_''test'+'.csv'))
    report = pd.concat([train_report, valid_report, test_report], axis=0, ignore_index=True)

    timeValue = torch.tensor(report.loc[:,report.columns.str.contains('\(')].values, dtype=torch.float32)
    config = [configEncode(eval(i)) for i in report.columns if '(' in i]
    config_Dict = {config[i]:i for i in range(len(config))}
    time_Dict = {report['File Name'][i]:timeValue[i] for i in range(len(report['File Name']))}
    return {'config_Dict':config_Dict, 'time_Dict':time_Dict}
    
if __name__ == '__main__':
    
    report_root = 'setcover_fixed_5fold'
    config_Dict, time_Dict = reportDict(report_root).values()
    print(config_Dict)
    