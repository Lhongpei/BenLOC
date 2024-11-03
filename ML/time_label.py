from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold,StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import logging
import os
from sklearn.metrics import mean_squared_error,accuracy_score
def preprocess(file):
    # 读取 CSV 文件
    data = pd.read_csv(file)

    # 找到所有要处理的列，排除包含 "feat" 和 "Name" 的列
    column = [col for col in data.columns if "feat" not in col and "Name" not in col]

    # 计算每行的最小值，并添加为 'min_time' 列
    data['min_time'] = data[column].min(axis=1)

    # id_vars 是不参与 melt 的列
    id_var = [col for col in data.columns if col not in column]

    # 将指定的列进行 melt 操作，转置为 'configuration' 和 'time' 列
    data = data.melt(id_vars=id_var, value_vars=column, var_name='configuration', value_name='time')

    # 创建一个初始值为0的 DataFrame，用于存储 feat_col 列
    config = pd.DataFrame(0, index=data.index, columns=[f"feat_{col}" for col in column])

    # 标记在原始数据中存在的 col 的位置
    for index, row in data.iterrows():
        col = row['configuration']  # 'configuration' 列中的值是原来的列名
        config.at[index, f'feat_{col}'] = 1  # 在对应的 feat_col 列中标记为 1

    # 将原始数据与标记后的配置列合并
    data = pd.concat([data, config], axis=1)

    return data

def baseline(file,default,col=None):
    data=pd.read_csv(file)
    default_time=shifted_geometric_mean(data[default],10)
    column=[coln for coln in data.columns if "feat" not in coln and "Name" not in coln]
    data['min_time']=data[column].min(axis=1)
    oracle=shifted_geometric_mean(data['min_time'],10)
    if col==None:
        default_time=shifted_geometric_mean(data[default],10)
        data=pd.read_csv(file)

        
        min_stf=100000000000
        best_stf=0
        best_col=None
        for col in column:
            sft_time=shifted_geometric_mean(data[col],10)
            print(col,sft_time)
            if sft_time<min_stf:
                best_stf=sft_time
                best_col=col
                min_stf=sft_time
                print(min_stf,best_col)
        return best_col,best_stf,default_time,oracle
    else:
        return shifted_geometric_mean(data[col],10),default_time,oracle


def process(data):
    columns = data.columns.tolist()
    data.columns = columns
    time_col=['File Name','time']
    feature_col=['File Name'] + [col for col in data.columns if 'feat' in col]
    time=data[time_col]
    feature=data[feature_col]
    X=get_X(feature)
    y=np.array(data['time'])
    return feature,time,X,y

def replace_inf_with_max(arr):
    max_val = np.finfo(np.float32).max
    min_val = np.finfo(np.float32).min
    arr[arr == -np.inf] = -1000
    arr[arr == np.inf] =3
    return arr
def get_X(feature):
    data=feature.drop(['File Name'],axis=1)
    X=np.array(data)
    X=replace_inf_with_max(X)

    return X

# def get_y(time):
#     y_time=np.concatenate((np.array(time['config_0']),np.array(time['config_1'] )))
#     y= np.log2((time['config_1'] + 1) / (time['config_0'] + 1))
#     y=np.array(y)
#     return y_time,y


def rfr_solver_obj(params, X, y):
    length = params.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42) 
    
    mse_scores = []
    for i in range(length):
        mse_fold = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # max_leaf_nodes = params.iloc[0]['max_leaf_nodes']
            # if max_leaf_nodes == 1:
            #     max_leaf_nodes = None
            rfr = RandomForestRegressor(
                n_estimators=int(params.iloc[i, 0]),
                max_depth=int(params.iloc[i, 1]),
                min_samples_leaf=int(params.iloc[i, 2]),
                min_samples_split=int(params.iloc[i, 3]),
                max_features=params.iloc[i, 4],
                # max_leaf_nodes=max_leaf_nodes,
                random_state=42, n_jobs=32)
            rfr.fit(X_train, y_train)
            y_pred = rfr.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)  # 计算 MSE
            mse_fold.append(mse)

        # 计算所有折的平均 MSE
        avg_mse = np.mean(mse_fold)
        mse_scores.append(avg_mse)
    return np.array(mse_scores)

def setup_logger(index, folder='time_label/log'):
    """配置每个进程的日志器，日志输出到指定的文件中"""
    if not os.path.exists(folder):
        os.makedirs(folder)

    log_file_path = os.path.join(folder, f'progress_{index}.log')

    logger = logging.getLogger(f'Process_{index}')
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def run_cross_validation(X_train, y_train,X_test,y_test,index,space):
    logger = setup_logger(index)
    try:

        # 创建 HEBO 优化器实例
        logger.info(f'Process_{index} start optimization')
        opt = HEBO(space)
        for i in range(400):  # 假设迭代1000次
            logger.info(f'Process_{index} iteration {i}')
            rec = opt.suggest(n_suggestions=1)
            mse_scores = rfr_solver_obj(rec, X_train, y_train)
            opt.observe(rec, mse_scores)

        best_params = pd.DataFrame(opt.X.iloc[[opt.y.argmin()]])
        # max_leaf_nodes = best_params.iloc[0]['max_leaf_nodes']
        # if max_leaf_nodes == 1:
        #     max_leaf_nodes = None

        rfr = RandomForestRegressor(
            n_estimators=int(best_params.iloc[0, 0]),
            max_depth=int(best_params.iloc[0, 1]),
            min_samples_leaf=int(best_params.iloc[0, 2]),
            min_samples_split=int(best_params.iloc[0,3]),
            max_features=best_params.iloc[0, 4],
            # max_leaf_nodes=max_leaf_nodes,
            random_state=42,
            n_jobs=32  # Limit joblib jobs inside other parallel routines
        )
        logger.info(f'Process_{index} start training')
        rfr.fit(X_train, y_train)
        y_predict_test = rfr.predict(X_test)
        y_predict_train= rfr.predict(X_train)
        logger.info(f'Process_{index} finished')
        return y_predict_train,y_predict_test, rfr, best_params

    except Exception as e:
        logger.error(f'An error occurred in process {index}: {e}', exc_info=True)
        # Depending on the application, you might want to re-raise the exception
        # or return a special value indicating failure.
        return None, None, None,None

def shifted_geometric_mean(metrics, shift):
    product_of_shifted_metrics=1
    for i in metrics:

        product_of_shifted_metrics =product_of_shifted_metrics*np.power(i+shift, 1/len(metrics))
        geom_mean_shifted =product_of_shifted_metrics  - shift
    
    return geom_mean_shifted
def time_train_f(time_train,y_predict,mode="shift"):
    rf_y_train_time= np.where(np.array(y_predict) > 0, time_train['config_0'], time_train['config_1'])
    rf_time=shifted_geometric_mean(rf_y_train_time,10) if mode=="shift" else rf_y_train_time.sum()
    time_0=shifted_geometric_mean(time_train['config_0'],10) if mode=="shift" else time_train['config_0'].sum()
    time_1=shifted_geometric_mean(time_train['config_1'],10) if mode=="shift" else time_train['config_1'].sum()
    time=time_1
    result = [np.min([time_train['config_0'].iloc[i], time_train['config_1'].iloc[i]]) for i in range(len(y_predict))]
    Oracle=shifted_geometric_mean(result,10)
    para=1
    if time_0<=time_1:
        time=time_0
        para=0
    Imp_stime=(time-rf_time)/time
    Imp_ub=(time-Oracle)/time
    return rf_time,time_1,time_0,Imp_stime,Oracle,Imp_ub,para

def time_test_f(time_test,y_predict,para,mode="shift"):
    rf_y_train_time= np.where(np.array(y_predict) > 0, time_test['config_0'], time_test['config_1'])
    rf_time=shifted_geometric_mean(rf_y_train_time,10) if mode=="shift" else rf_y_train_time.sum()
    time_0=shifted_geometric_mean(time_test['config_0'],10) if mode=="shift" else time_test['config_0'].sum()
    time_1=shifted_geometric_mean(time_test['config_1'],10) if mode=="shift" else time_test['config_0'].sum()
    time=shifted_geometric_mean(time_test[f'config_{para}'],10) if mode=="shift" else time_test[f'config_{para}'].sum()
    Imp_stime=(time-rf_time)/time

    result = [np.min([time_test['config_0'].iloc[i], time_test['config_1'].iloc[i]]) for i in range(len(y_predict))]
    Oracle=shifted_geometric_mean(result,10)
    Imp_ub=(time-Oracle)/time
    return rf_time,time_1,time_0,Imp_stime,Oracle,Imp_ub
def accuracy_f(time,y_predict):
    y_label= [0 if a < b else 1 for a, b in zip(time['config_0'], time['config_1'])]
    y_label=np.array(y_label)
    rf_y_label=np.where(np.array(y_predict) > 0, 0, 1)
    pertrue0=list(y_label).count(0)/len(y_label)
    pertrue1=list(y_label).count(1)/len(y_label)
    per0=list(rf_y_label).count(0)/len(rf_y_label)
    per1=list(rf_y_label).count(1)/len(rf_y_label)
    accuracy = accuracy_score(y_label, rf_y_label)
    return pertrue0,pertrue1,per0,per1,accuracy
# def get_result(X_train,X_test,time_train,time_test,rf):
#     col=['RF_time','AlwaysLC_time','NeverLC_time','Imp_stime','Imp_ub','Oracle','accuracy','0%','1%','0_predict','1_predict','RF_sum','AlwaysLC_sum','NeverLC_sum','Imp_sum','Oracle_sum','Imp_ub_sum']
#     df = pd.DataFrame(columns=col)
#     row=[]

#     y_train_predict=rf.predict(X_train)
#     n = y_train_predict.shape[0] // 2
#     y_train_predict = y_train_predict.reshape(2, n).T
#     y_train_predict=np.log2((y_train_predict[:,1] + 1) / (y_train_predict[:,0]  + 1))
#     y_test_predict=rf.predict(X_test)
#     n = y_test_predict.shape[0] // 2
#     y_test_predict = y_test_predict.reshape(2, n).T
#     y_test_predict=np.log2((y_test_predict[:,1] + 1) / (y_test_predict[:,0]  + 1))
#     rf_time,time_1,time_0,Imp_stime,Oracle,Imp_ub,para_sft=time_train_f(time_train,y_train_predict,mode="shift")
#     pertrue0,pertrue1,per0,per1,accuracy=accuracy_f(time_train,y_train_predict)
#     rf_sum,sum_1,sum_0,Imp_sum,Oracle_sum,Imp_ub_sum,para_sum=time_train_f(time_train,y_train_predict,mode="sum")
#     row=[rf_time,time_1,time_0,Imp_stime,Oracle,Imp_ub,accuracy,pertrue0,pertrue1,per0,per1,rf_sum,sum_1,sum_0,Imp_sum,Oracle_sum,Imp_ub_sum]#,rf_node,node_1,node_0,Imp_node]
#     df.loc['train'] = row 
#     rf_time,time_1,time_0,Imp_stime,Oracle,Imp_ub=time_test_f(time_test,y_test_predict,para_sft,mode="shift")
#     pertrue0,pertrue1,per0,per1,accuracy=accuracy_f(time_test,y_test_predict)
#     rf_sum,sum_1,sum_0,Imp_sum,Oracle_sum,Imp_ub_sum=time_test_f(time_test,y_test_predict,para_sum,mode="sum")
#     row=[rf_time,time_1,time_0,Imp_stime,Oracle,Imp_ub,accuracy,pertrue0,pertrue1,per0,per1,rf_sum,sum_1,sum_0,Imp_sum,Oracle_sum,Imp_ub_sum]#,rf_node,node_1,node_0,Imp_node]
#     df.loc['test'] = row 
#     return df
def get_result(oracle,default,local_min,predict,baseline_col):
    baseline_name=baseline_col
    default_time=default
    baseline=local_min
    ipv_default=(default-predict)/default
    ipv_baseline=(baseline-predict)/baseline
    ipv_oracle=(default-oracle)/default
    return [baseline_name,default_time,baseline,predict,ipv_default,ipv_baseline,oracle,ipv_oracle]



def save_result(y_label,time,rf_y_label,y_predict,rf_y_test_time,y,pred_time):
    n=pred_time.shape[0]//2

    temp_res_train = pd.DataFrame({
    'File Name': time['File Name'],
    'config':y_label,
    'config_0':time['config_0'],
    'config_1':time['config_1'],
    'y':y,
    'pred_confgi':rf_y_label,
    'pred_y':y_predict,
    'pred_time':rf_y_test_time,
    'pred_time0':pred_time[:n],
    'pred_time1':pred_time[n:]

    })
    return temp_res_train

if __name__ == "__main__":
    space = DesignSpace().parse([
        {'name': 'n_estimators', 'type': 'int', 'lb': 300, 'ub': 500},
        {'name': 'max_depth', 'type': 'int', 'lb': 3, 'ub': 10},
        {'name':  'min_samples_leaf', 'type': 'int', 'lb': 3, 'ub': 20},
        {'name':'min_samples_split','type': 'int', 'lb': 4, 'ub': 20},
        {'name': 'max_features', 'type': 'cat', 'categories': ['sqrt', 'log2', None]},
        #{'name': 'max_leaf_nodes', 'type': 'int', 'lb': 1, 'ub': 4000}

        #{'name':'criterion','type':'cat','categories':['squared_error', 'poisson', 'friedman_mse', 'absolute_error']}
    ])
    #for index in ['indset','load_balance','nn_verification','setcover']:
    default='MipLogLevel-2'
    for index in ['load_balance','indset']:
        if index=="load_balance":
            ind_range=range(7,11)
        else:
            ind_range=range(1,11)
        # if index=='setcover':
        #     ind_range=[2]
        # else:
        #     ind_range=[1,2]
        for i in ind_range:
            train_file=os.path.join(index,f'fold_{i}_train.csv')
            test_file=os.path.join(index,f'fold_{i}_test.csv')
            data_train=preprocess(train_file)
            data_train.to_csv(f"train_{index}_fold{i}.csv")
            data_test=preprocess(test_file)
            feature_train,time_train,X_train,y_train=process(data_train)
            feature_test,time_test,X_test,y_test=process(data_test)
            results=run_cross_validation(X_train, y_train,X_test,y_test,f'{index}_fold{i}',space)
            folder_path = 'time_label/model_result/'

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(f'time_label/model_result/model_{index}_fold{i}.pickle', 'wb') as f:
                pickle.dump(results, f)
            y_predict_train,y_predict_test, rfr, best_params=results
            print(y_predict_train)
            if y_predict_train is None:
                continue
            data_train['predict_y']=y_predict_train
            data_test['predict_y']=y_predict_test
            min_idx_train = data_train.groupby('File Name')['predict_y'].idxmin()
            
            print(min_idx_train)
            result_train = data_train.loc[min_idx_train].reset_index(drop=True)
            min_idx_test = data_test.groupby('File Name')['predict_y'].idxmin()
            result_test = data_test.loc[min_idx_test].reset_index(drop=True)
            best_col,best_stf,default_time,oracle=baseline(train_file,default)
            row_train=get_result(oracle,default_time,best_stf,shifted_geometric_mean(result_train['time'],10),best_col)
            best_stf,default_time,oracle=baseline(test_file,default,best_col)
            row_test=get_result(oracle,default_time,best_stf,shifted_geometric_mean(result_test['time'],10),best_col)
            col=['baseline_name','default_time','baseline_time','rf_time','ipv_default','ipv_baseline','oracle','ipv_oracle']
            df = pd.DataFrame(columns=col)        
            df.loc['train']=row_train
            df.loc['test']=row_test
            folder_path = 'time_label/stat_result/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            folder_path = 'time_label/pred_result/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            df.to_csv(f"time_label/stat_result/stat_{index}_fold{i}.csv")

            data_test.to_csv(f"time_label/pred_result/pred_{index}_fold{i}_test.csv")

            data_train.to_csv(f"time_label/pred_result/pred_{index}_fold{i}_train.csv")

            # rf=results[1]
            # df=get_result(X_train,X_test,time_train,time_test,rf)
            # df.to_csv(f"time_label/stat_result/stat_{index}_fold{i}.csv")
            # y_train_predict=rf.predict(X_train)
            # pred_time_train=y_train_predict
            # y_test_predict=rf.predict(X_test)
            # pred_time_test=y_test_predict
            # y_train_predict=rf.predict(X_train)
            # n = y_train_predict.shape[0] // 2
            # y_train_predict = y_train_predict.reshape(2, n).T
            # y_train_predict=np.log2((y_train_predict[:,1] + 1) / (y_train_predict[:,0]  + 1))
            # y_test_predict=rf.predict(X_test)
            # n = y_test_predict.shape[0] // 2
            # y_test_predict = y_test_predict.reshape(2, n).T
            # y_test_predict=np.log2((y_test_predict[:,1] + 1) / (y_test_predict[:,0]  + 1))
            # y_label= [0 if a < b else 1 for a, b in zip(time_train['config_0'], time_train['config_1'])]
            # y_label=np.array(y_label)
            # rf_y_label=np.where(np.array(y_train_predict) > 0, 0, 1)
            # rf_y_train_time= np.where(np.array(y_train_predict) > 0, time_train['config_0'], time_train['config_1'])
            # result_train=save_result(y_label,time_train,rf_y_label,y_train_predict,rf_y_train_time,y_train,pred_time_train)
            # result_train.to_csv(f"time_label/pred_result/pred_{index}_fold{i}_train.csv")
            # y_label= [0 if a < b else 1 for a, b in zip(time_test['config_0'], time_test['config_1'])]
            # y_label=np.array(y_label)
            # rf_y_label=np.where(np.array(y_test_predict) > 0, 0, 1)
            # rf_y_test_time= np.where(np.array(y_test_predict) > 0, time_test['config_0'], time_test['config_1'])
            # result_test=save_result(y_label,time_test,rf_y_label,y_test_predict,rf_y_test_time,y_test,pred_time_test)
            # result_test.to_csv(f"time_label/pred_result/pred_{index}_fold{i}_test.csv")
        