import os

from omegaconf import OmegaConf


def create_config(root_path=None):
    
    # 创建一个配置字典
    config = {
        "fold": 5,
        "hyper_L2R": {
            "alpha": 5,
            "lr": 1e-3,
            "epoch": 500,
            "test_epoch": 1,
            "batchsize": 32,
            "stepsize": 150,
            "gamma": 0.1,
            "pooling": "add",
            "tot_config_dim": 25,
            "ranklist":{
                "config_dim":25
            },
            "regression":{
                "config_dim":10
            }
        },
        "encoder": {
            "input_dim_xs": 5,
            "input_dim_xt": 8,
            "input_dim_edge": 1,
            "num_layers": 1,
            "hidden_dim": 80,
            "mlp_hidden_dim": 90,
            "enc_dim": 40
        },
        "predict_L2R": {
            "hidden_dim": 80,
            "mlp_hidden_dim": [200,100,50],
            "num_layers": 3
        },
        "pyg_regression": {
            "gnn_hidden_dim": 50,
            "gnn_out_dim": 30,
            "pred_mlp_hidden_dim": [400,200,100,50,25],
            "config_mlp_hidden_dim": [400,200,100,50],
            "config_mlp_layers": 1, 
            "pred_mlp_layers": None,
            "num_layers": 3,
            "task_dim": 10,
            "pooling": "add",
            "jk_mode": "lstm",
            "GNN": "GraphSAGE"
        },
        "pyg_ranklist": {
            "gnn_hidden_dim": 300,
            "gnn_out_dim": 100,
            
            "pooling_mlp_hidden_dim": [200,300,100],
            "pooling_mlp_layers": 1,
            "pooling_mlp_out_dim": 100,
            
            "mlp_hidden_dim": [100,200,50],
            "pred_mlp_layers": None,
            "num_layers": 3,
            
            "task_dim": 25,
            "pooling": "add",
            "jk_mode": "cat",
            "GNN": "GAT"
        },
        "pyg_ranklist_node": {
            "gnn_hidden_dim": 300,
            "gnn_out_dim": 1,
            "num_layers": 3,
            
            "task_dim": 25,
            "jk_mode": "cat",
            "GNN": "GraphSAGE"
        },
        "seed": 32,
        "cpu_threads": 64,
        "paths": {
            "setcover": {
                "data_folder": "/data/ml_tuner/setcover-flat",
                "store_folder": os.path.join(root_path, "datasets/setcover"),
            },
            "indset": {
                "data_folder": "/data/ml_tuner/indset-flat",
                "store_folder": os.path.join(root_path, "datasets/indset"),
            },
            "mix_set_ind": {
                "data_folder": '/data/ml_tuner/mix_set_ind',
                "store_folder": os.path.join(root_path, "datasets/mix_set_ind"),
            },
            "miplib": {
                "data_folder": '/data/MIPLIB-2017-Collection-COPT71RC-MipPres',
                "store_folder": os.path.join(root_path, "datasets/miplib"),
            },
            "predict_config_folder": os.path.join(root_path, "setcover_config"),
            "savedEncoder": os.path.join(root_path, "setcover_train/encoder100.pth"),
            "setcover_folder": os.path.join(root_path, "setcover-flat"),
            "setcover_report_root": os.path.join(root_path, "labels/balanced_setcover"),
            "indset_folder": os.path.join(root_path, "indset-flat"),
            "indset_report_root": os.path.join(root_path, "labels/indset_fixed_5fold"),
            "train_folder": os.path.join(root_path, "fold_rank_train"),
            "result_folder": os.path.join(root_path, "fold_rank_result"),
            "mix_set_int_folder": os.path.join(root_path, "mix_set_ind"),
            "mixed_report_root": os.path.join(root_path, "labels/balanced_setcover")
        }
    }
    if not os.path.exists("Config"):
        os.makedirs("Config")
    # 使用 OmegaConf 将配置转换为 YAML 字符串并保存到文件
    config_path = os.path.join(root_path, "src/config/config.yaml")
    OmegaConf.save(config, config_path)
    return config_path
if __name__ == '__main__':
    create_config('repo/src/')