# ML4MOC: A Benchmark for Optimizer Configuration using Machine Learning

## Datasets

Presolved Data is stored in `.\instance`.

## Handcraft Feature Extraction

folder: extract_feature

- Static feature extraction for MIP problems: extract_problem.py
- Extract other features from COPT solution log: extract_log_feature.py
- Feature combination and preprocessing: combine.py

## Train Random Forest

## Train GNNs-based Predict Model

```bash
python DL/train_gnn_predictor.py \
    --use_wandb True \
    --modeGNN GAT \
    --fold 2 \
    --reproData True \ # If you want to reload data
    --default_index 7 \
    --report_root_path /your/path/to/labels \
    --problem_root_path /your/path/to/instances \
    --result_root_path /your/path/to/save_result \
    --save_model_path /your/path/to/save_model \
    --alpha 0.001 \
    --lr 0.001 \
    --epoch 100 \
    --batchsize 32 \
    --stepsize 10 \
    --gamma 0.9

```

## Train VGAE\GAE and Predict

## Reference
