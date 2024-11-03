from MIPmodel import MIPmodel
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_problem(file_path,output):
    allpath=os.listdir(file_path)
    model=MIPmodel()
    norm=pd.DataFrame()
    for p in tqdm(allpath):
        ind,attr,x_s,x_t,temp = model.generStatic(file_path=os.path.join(file_path,p))
        norm=pd.concat([norm,temp],ignore_index=True)
    norm.insert(0,'File Name',pd.Series(allpath))
    print(norm.shape)
    norm=norm.fillna(0)
    norm.to_csv(output,index=False)