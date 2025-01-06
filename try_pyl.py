from ml4moc import ML4MOC, TabParams, TabNetRegressor, TabNet
from sklearn.ensemble import RandomForestRegressor
params = TabParams(default="MipLogLevel-2", label_type="log_scaled", shift_scale=10)
model = ML4MOC(params)
model.load_dataset("indset",processed=True)
tabmodel = TabNetRegressor(n_d = 5, n_a = 5)
tabmodel.fit(model.get_X, model.get_Y, max_epochs=1)
model.set_trainner(tabmodel.network)
model.train_test_split_by_splitfile('./ml4moc/data/fold_name/indset_fold_2.pkl')
model.fit()
evaluation_results=model.evaluate()
print(evaluation_results)