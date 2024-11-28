# ML4MOC: A Benchmark for Optimizer Configuration using Machine Learning

<img src="pic/workflow.png" alt="WorkFlow" width="800">

## Datasets

See `ml4moc/data/dataset.md` for the introduction.

## ML4MOC Class Documentation

The `ML4MOC` class is designed to facilitate the training and evaluation of machine learning models for optimization problems. It provides methods for data preprocessing, model selection, training, and evaluation. Below is a step-by-step guide on how to use the class:

---

### 1. **Setting Parameters**

The parameters for the `ML4MOC` class can be set during initialization. You can either pass a `Params` object or use the default parameters.

```python
from ml4moc.params import Params
params = Params(default="MipLogLevel-2", label_type="log_scaled", shift_scale=10)
model = ML4MOC(params)
```

- **label_type**: Specifies whether the labels are scaled (`log_scaled`) or in their original form (`original`).
- **default**: The column name representing the default time in the dataset.
- **shift_scale**: The scale used for the geometric mean during the baseline calculation.

---

### 2. **Setting the Learner (Model)**

You can set a model (e.g., `RandomForestRegressor`) as the trainer for the `ML4MOC` class using the `set_trainner` method.

```python
from sklearn.ensemble import RandomForestRegressor
model.set_trainner(RandomForestRegressor(verbose=1))
```

- **RandomForestRegressor**: A random forest regressor model is used as the default estimator for the training process. You can also use other sklearn estimators, such as `LinearRegression`, `SVR`, etc. 
- If you want to adapt customized estimator, please provide APIs of `model.fit()` and `model.predict()` just like sklearn's APIs.

---

### 3. **Setting the Dataset**

You can set our datasets or input your datasets.

#### Loading Provided labeled Datasets

You can load a dataset using the `load_dataset` method. The dataset will include both features and labels, and can be processed optionally.

```python
model.load_dataset("setcover", processed=True, verbose=True)
```

- **dataset**: The dataset to be loaded (e.g., `setcover`, `indset`, `nn_verification`, etc.).
- **processed**: If set to `True`, it loads the processed version of the dataset.
- **verbose**: If set to `True`, it prints information about the dataset.

After loading the dataset, the `feat` (features) and `label` (labels) are stored, and preprocessing can be applied as needed.

#### Input Datasets

```python
model.set_train_data(Feature, Label)
#model.set_test_data(Test_Feature, Test_Label) (OPTIONAL)
```

Both `Feature` and `Label` should be the type of `pandas.DataFrame`. Please follow our formats!

*If you want use `model.evaluation`, you can either set test datasets or only set train datasets and use train-test-split.*

---

### 4. **Train-Test Split**

*If you have set test dataset, please skip this step.*

To split the dataset into training and testing sets, you can use the `train_test_split` method.

```python
model.train_test_split(test_size=0.2)
```

- **test_size**: Fraction of the data to be used as the test set (e.g., `0.2` means 20% for testing, 80% for training).

This will divide the features and labels into training and testing sets and preprocess the data accordingly.

---

### 5. **Fitting the Model**

Once the dataset is loaded and the model is set, you can fit the model using the `fit` method.

```python
model.fit()
```

The `fit` method will train the model using the processed features (`X`) and labels (`Y`).

- The model is trained on the processed training data using the `trainner` model set earlier.

---

### 6. **Evaluating the Model**

After training, you can evaluate the model using the `evaluate` method. This method will make predictions on both the training and testing datasets and return a DataFrame with evaluation results.

```python
evaluation_results = model.evaluate()
```

- **evaluation_results**: A DataFrame containing the evaluation results, including the baseline, default time, model prediction time, and oracle performance for both training and testing datasets.

The evaluation method returns the following columns:

- **baseline_name**: Name of the baseline method used.
- **default_time**: Default time using the baseline method.
- **baseline_time**: Time predicted by the baseline method.
- **rf_time**: Time predicted by the random forest model.
- **ipv_default**: The improvement in performance relative to the baseline (train).
- **ipv_baseline**: The improvement in performance relative to the baseline (test).
- **oracle**: The optimal oracle time.
- **ipv_oracle**: The improvement relative to the oracle.

---

### **Estimator Parameters Selection** (Optional)

we support use sklearn's model-selection methods to select parameters.

```python
model.cross_validation(CVer, parameters_space log_name)
```

if you use `RandomForest`, `LightGBM` or `GBDT`, we will support our parameters space as default.

Use following commands to check our parameters space.

```python
model.rfr_parameter_space
model.lgbm_parameter_space
model.gbdt_parameter_space
```

### Example Usage Flow

```python
# Step 1: Initialize the model with parameters
from ml4moc.params import Params
params = Params(default="MipLogLevel-2", label_type="log_scaled", shift_scale=10)
model = ML4MOC(params)

# Step 2: Set the machine learning model (e.g., Random Forest)
from sklearn.ensemble import RandomForestRegressor
model.set_trainner(RandomForestRegressor(verbose=1))

# Step 3: Load the dataset and preprocess
model.load_dataset("setcover", processed=True, verbose=True)

# Step 4: Split the dataset into training and test sets
model.train_test_split(test_size=0.2)

# Step 5: Train the model
model.fit()

# Step 6: Evaluate the model and get results
evaluation_results = model.evaluate()

# Output evaluation results
print(evaluation_results)
```
