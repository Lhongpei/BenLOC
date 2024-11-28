from ml4moc import ML4MOC, Params
from sklearn.ensemble import RandomForestRegressor
params = Params(default="MipLogLevel-2", label_type="log_scaled", shift_scale=10)
model = ML4MOC(params)
model.load_dataset("setcover")
model.set_trainner(RandomForestRegressor(verbose=1))
model.train_test_split()
model.fit()
model.evaluate()