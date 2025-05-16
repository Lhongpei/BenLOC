import os

from omegaconf import OmegaConf


def create_config(root_path=None):
    
    # 创建一个配置字典
    config = {
        "fold": 5,
        "predict_task": {
            "alpha": 5,
            "epoch": 1500,
            "test_epoch": 1,
            "batchsize": 128,
            "steplr":{
                "lr": 0.001,
                "stepsize": 10,
                "gamma": 0.98,
            },
            "cosinelr":{
                "warmup_epoch": 50,
                "final_lr": 1e-5,
                "base_lr": 1e-2,
            },
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
        "nn_config": {
            "gnn_hidden_dim": 64,
            #"gnn_out_dim": 64,
            "heads": 4,
            "feat_mlp_hidden_dim": [128,256, 512, 256],
            "feat_mlp_layers": 1,
            "feat_mlp_out_dim": 100,
            
            "pooling_mlp_hidden_dim": [128,256],
            "pooling_mlp_layers": 1,
            "pooling_mlp_out_dim": 128,
            
            "pred_mlp_hidden_dim": [128,32],
            "pred_mlp_layers": None,
            "num_layers": 3,
            
            "task_dim": 2,
            "pooling": "add",
            "jk_mode": "lstm",
            "GNN": "GAT",
            "dropout": 0.2,
        },
        "seed": 32,
        "cpu_threads": 64,
    }
    if not os.path.exists("Config"):
        os.makedirs("Config")
    # 使用 OmegaConf 将配置转换为 YAML 字符串并保存到文件
    config_path = os.path.join(root_path, "src/config/config.yaml")
    OmegaConf.save(config, config_path)
    return config_path
if __name__ == '__main__':
    create_config('repo/src/')