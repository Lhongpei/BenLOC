# Introduction to Datasets

Presolved Data is stored in `.\instance`. The folder structure after the datasets are set up looks as follows

```bash
instances/
  MIPLIB/                   -> 1065 instances
  set_cover/                -> 3994 instances
  independent_set/          -> 1604 instances
  nn_verification/          -> 3104 instances
  load_balancing/           -> 2286 instances
```

### Dataset Description

#### MIPLIB

Heterogeneous dataset from [MIPLIB 2017](https://miplib.zib.de/), a well-established benchmark for evaluating MILP solvers. The dataset includes a diverse set of particularly challenging mixed-integer programming (MIP) instances, each known for its computational difficulty. 

#### Set Covering

This dataset consists of instances of the classic Set Covering Problem, which can be found [here](https://github.com/ds4dm/learn2branch/tree/master). Each instance requires finding the minimum number of sets that cover all elements in a universe. The problem is formulated as a MIP problem. 

#### Maximum Independent Set

This dataset addresses the Maximum Independent Set Problem, which can be found [here](https://github.com/ds4dm/learn2branch/tree/master). Each instance is modeled as a MIP, with the objective of maximizing the size of the independent set. 

#### NN Verification

This “Neural Network Verification” dataset is to verify whether a neural network is robust to input perturbations can be posed as a MIP. The MIP formulation is described in the paper [On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models (Gowal et al., 2018)](https://arxiv.org/abs/1810.12715). Each input on which to verify the network gives rise to a different MIP. 

#### Load Balancing

This dataset is from [NeurIPS 2021 Competition](https://github.com/ds4dm/ml4co-competition). This problem deals with apportioning workloads. The apportionment is required to be robust to any worker’s failure. Each instance problem is modeled as a MILP, using a bin-packing with an apportionment formulation.

### Dataset Spliting

Each dataset was split into a training set  $D_{\text{train}}$ and a testing set $D_{\text{test}}$, following an approximate 80-20 split. Moreover, we split the dataset by time and "optimality", which means according to the proportion of optimality for each parameter is similar in training and testing sets. This ensures a balanced representation of both temporal variations and the highest levels of parameter efficiency in our data partitions.

To split the datasets and create different folds for cross validation, run

```bash
   python extract_feature/split_fold.py \
       --dataset_name "your_dataset_name" \
       --time_path "/your/path/to/soving times" \
       --feat_path "/your/path/to/features" \
   ``` 

## Handcraft Feature Extraction

**Folder:** ```extract_feature```

**Problem format:** ```your_file.mps.gz``` or ```your_file.lp```

**Log format:** ```your_file.log```

1. Static feature extraction for MIP problems: run

   ```bash
   python extract_feature/extract_problem.py \
       --problem_folder "/your/path/to/instances" \
       --dataset_name "your_dataset_name" \
   ```

2. Extract other features from COPT solution log: run

   ```bash
   python extract_feature/extract_log_feature.py \
       --log_folder "/your/path/to/solving logs" \
       --dataset_name "your_dataset_name" \
   ```

3. Feature combination and preprocessing: run

   ```bash
   python extract_feature/combine.py \
       --dataset_name "your_dataset_name" \
   ```
4. Label extraction from COPT solution log: run

   ```bash
   python extract_feature/extract_time.py \
       --log_folder "/your/path/to/solving logs" \
       --dataset_name "your_dataset_name" \