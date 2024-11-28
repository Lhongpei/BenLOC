from ml4moc import ML4MOC, baseline
from sklearn.ensemble import RandomForestRegressor
model = ML4MOC()
model.load_dataset("setcover")
baseline(model.label, model.default)
model.set_trainner(RandomForestRegressor())
# model.fit()