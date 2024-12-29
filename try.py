from ml4moc import ML4MOC, MLParams
from sklearn.ensemble import RandomForestRegressor
params = MLParams(default="MipLogLevel-2", label_type="log_scaled", shift_scale=10)
model = ML4MOC(params)
model.load_dataset("indset",processed=True)
model.set_trainner(RandomForestRegressor(verbose=1))
model.train_test_split_by_splitfile('./ml4moc/data/fold_name/indset_fold_1.pkl')
# model.fit()
# evaluation_results=model.evaluate()