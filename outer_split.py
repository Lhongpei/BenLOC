from ml4moc import ML4MOC, TabParams, TabNetRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
params = TabParams(default="MipLogLevel-2", label_type="log_scaled", shift_scale=10)
model = ML4MOC(params)
model.load_dataset("nn_verification",processed=True)
feat, label = model.feat, model.label

split = pickle.load(open('./ml4moc/data/fold_name/nn_verification_fold_2.pkl', 'rb'))
train_feat = feat[feat['File Name'].isin(split['train'])]
test_feat = feat[feat['File Name'].isin(split['test'])]
test_label = label[label['File Name'].isin(split['test'])]
train_label = label[label['File Name'].isin(split['train'])]
model.set_train_data(train_feat, train_label)
model.set_trainner(TabNetRegressor(n_d = 32, n_a = 32))
# model.set_trainner(RandomForestRegressor())
# model.train_test_split_by_splitfile('./ml4moc/data/fold_name/indset_fold_2.pkl')
model.fit(patience=100, max_epochs=1000)
model.set_test_data(test_feat, test_label)
evaluation_results=model.evaluate()
print(evaluation_results)