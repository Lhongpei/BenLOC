import coptpy as cp
from itertools import product
from scipy import sparse as sp
import torch
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import statistics
import math
import networkx as nx
from scipy.stats import describe

class MIPmodel:
    def __init__(self,name='default',file_path=None,log=False):
        self.env=cp.Envr()
        self.model=self.env.createModel(name)
        self.baseline=None
        self.hasRead=False
        self.log=log
        self.resultLis=["SolvingTime","Nodecnt","BestGap"]
        self.paraLis=['CutLevel', 'RootCutLevel', 'TreeCutLevel', 'RootCutRounds', 'NodeCutRounds', 'HeurLevel', 'RoundingHeurLevel', 'DivingHeurLevel', 
        'SubMipHeurLevel', 'StrongBranching', 'ConflictAnalysis', 'MipStartMode', 'MipStartNodeLimit']
        self.paraRange= {
        "CutLevel": [0, 1, 2, 3],
        "RootCutLevel": [0, 1, 2, 3],
        "TreeCutLevel": [0, 1, 2, 3],
        "HeurLevel": [0, 1, 2, 3],
        "RoundingHeurLevel": [0, 1, 2, 3],
        "DivingHeurLevel": [0, 1, 2, 3],
        "SubMipHeurLevel": [0, 1, 2, 3],
        "StrongBranching": [0, 1, 2, 3],
        }
        self.lowLevelSelect = ["CutLevel", "HeurLevel"]
        self.HignLevelSelect=["RootCutLevel","TreeCutLevel","RoundingHeurLevel","DivingHeurLevel","SubMipHeurLevel","StrongBranching"]
        self.CutHighLevelSelect=["RootCutLevel","TreeCutLevel"]
        self.HeurHighLevelSelect=["RoundingHeurLevel","DivingHeurLevel","SubMipHeurLevel","StrongBranching"]
        if self.log:
            self.model.setParam('Logging',1)
        else:
            self.model.setParam('Logging',0)
        if not file_path is None:
            self.read(file_path) 
            
    def read(self,file_path):
        """
        用于根据文件路径读入模型
    
        参数:
        file_path:文件路径
        """
        self.model.read(file_path)
        self.baseline=None
        self.hasRead=True
        
    def solve(self):
        """_summary_ 用来求解MIP问题
        """
        if self.hasRead:
            self.model.solve()
        else:
            raise ResourceWarning('Hasn\'t define MIP problem')
        
    def resetParam(self):
        self.model.resetParam()
        if self.log==False:
            self.model.setParam('Logging',0)
        
    def runbaseline(self):
        """_summary_ 用来求解默认参数设置下的结果属性
        """
        self.resetParam()
        self.solve()
        self.baseline=self.showResult()
        for i in self.resultLis:
            self.baseline[i]=self.model.getAttr(i)
            
    def getBaseline(self):
        """
        得到在默认参数设置下的结果属性
        """
        if self.baseline is None:
            self.runbaseline()
        return self.baseline
    
    def setParamByDict(self,paraDict):
        for i in paraDict:
            self.model.setParam(i,paraDict[i])
            
    def showResult(self):
        """_summary_ 用来显示结果属性

        Returns:
            dict: 返回一个字典，键为结果属性名称，值为结果属性值
        """
        self.solve()
        tmp={}
        for i in self.resultLis:
            tmp[i]=self.model.getAttr(i)
        return tmp
    
    def testParam(self,paraDict):
        """_summary_ 用来测试参数设置下的结果属性

        Args:
            paraDict (字典): 该字典的键为参数名，值为参数值

        Returns:
            tuple: 返回元组，第一个元素为求解时间，第二个元素为节点数，第三个元素为最优解的gap.
        """        
        self.setParamByDict(paraDict)
        result=self.showResult()
        return result
    
    def setTimeLimit(self,time):
        """_summary_ 用来设置求解时间上限

        Args:
            time (int): 求解时间上限
        """
        self.model.setParam('TimeLimit',time)
        
    def getParamName(self):
        """_summary_ 用来得到所有参数的名称

        Returns:
            list: 返回所有参数的名称
        """
        return self.paraLis
    
    def setParamByLis(self,setLis):
        """_summary_ 用列表来设置参数

        Args:
            paraLis (list): 参数值的列表
        """
        assert len(setLis)==len(self.paraLis)
        dict=zip(self.paraLis,setLis)
        self.setParamByDict(dict)
    
    def getParam(self):
        """_summary_ 用来得到所有参数的值

        Returns:
            dict: 返回一个字典，键为参数名，值为参数值
        """
        tmp={}
        for i in self.paraLis:
            tmp[i]=self.model.getAttr(i)
        return tmp
    
    def bruteFindAll(self,paraSubList,timelimit=10000,accelerate=False):
        """_summary_ 用来暴力搜索参数设置下的结果属性

        Args:
            paraSubList (list): 选择属性的列表
            accelerate (bool, optional): 是否加速. Defaults to False, 注意只有在只想要最优解的情况下才能加速
        Returns:
            dict: 返回一个字典，键为参数设置，值为结果属性
        """
        subset_parameters = {param: self.paraRange[param] for param in paraSubList}

        parameter_combinations = list(product(*subset_parameters.values()))
        parameters_list = []
        for combination in parameter_combinations:
            parameters_dict = {key: value for key, value in zip(subset_parameters.keys(), combination)}
            parameters_list.append(parameters_dict)
        resultForParams={}
        for params in parameters_list:
            params_tuple = tuple(params.items())  # Convert the dictionary to a tuple of key-value pairs
            self.setTimeLimit(timelimit)
            result=self.testParam(params)
            #判断是否加速
            if accelerate and result['SolvingTime']<timelimit and result['BestGap']<1e-8:
                timelimit=result['SolvingTime']
                self.setTimeLimit(timelimit)
                
            resultForParams[params_tuple] = result
        return resultForParams
    
    def bruteMinTime(self,paraSubList,timelimit=10000,accelerate=False):
        """_summary_ 用来暴力搜索参数设置下的最小求解时间

        Args:
            paraSubList (list): 选择属性的列表
        Returns:
            tuple: 返回一个元组，第一个元素为最小求解时间，第二个元素为参数设置
        """
        baseline=self.getBaseline()
        timelimit=baseline['SolvingTime']+0.5
        print('Baseline:',baseline)
        result=self.bruteFindAll(paraSubList,accelerate=accelerate,timelimit=timelimit)
        minTime=100000
        minParams=None
        for i in result:
            if result[i]['SolvingTime']<minTime and result[i]['BestGap']<1e-8:
                minTime=result[i]['SolvingTime']
                minParams=i
        return minTime,minParams
    
    def searchAll(self,paraSubList,gap,timelimit=1000):
        """_summary_ 搜索所有参数设置下的结果属性
        
        return: 列表，第一个元素为求解时间，第二个元素为参数,第三个元素为最优解的gap，第四个元素为baseline
        """
        baseline=self.getBaseline()
        subset_parameters = {param: self.paraRange[param] for param in paraSubList}

        parameter_combinations = list(product(*subset_parameters.values()))
        parameters_list = []
        for combination in parameter_combinations:
            parameters_dict = {key: value for key, value in zip(subset_parameters.keys(), combination)}
            parameters_list.append(parameters_dict)
        resultForParams={}
        for params in parameters_list:
            params_tuple = tuple(params.items())  # Convert the dictionary to a tuple of key-value pairs
            self.setTimeLimit(timelimit)
            result=self.testParam(params)
            
            #if result['SolvingTime']<timelimit and result['BestGap']< gap:
             #   timelimit=result['SolvingTime']
             #   self.setTimeLimit(timelimit)
                
            resultForParams[params_tuple] = result
            index=[]
            solvingtime=[]
            basel=[]
            bestgap=[]
            for i in resultForParams:
                index.append(i)
                solvingtime.append(resultForParams[i]['SolvingTime'])
                basel.append(baseline)
                bestgap.append(resultForParams[i]['BestGap'])
        return solvingtime,index,bestgap,basel
    
    def findAllPro(self,maxTime,minTime,paraSubList,file_path,report_path=None):
        """_summary_ 用来找到所有参数设置下的结果属性
        """
        index_set=[]
        if not report_path is None:
            allreport=os.listdir(report_path)
            choReport_path=os.path.join(report_path,allreport[0])
            df=pd.read_csv(choReport_path)
            for index,i in enumerate(df['Time']):
                if i<maxTime and i>minTime:
                    index_set.append(index)
        allpro=os.listdir(file_path)
        
        
        #proLis=[allpro[self.findFileName(file_path,df['Name'][i]+'.mps.gz')]for i in index_set]
        proLis=allpro
        allTime=[]
        allAttri=[]
        bestgap=[]
        basetime=[]
        basegap=[]
        for i in tqdm(proLis,desc='Solving process:'):
            print('Solving: '+i)
            choReport_path=os.path.join(file_path,i)
            self.read(choReport_path)
            time,attri,gap,bl=self.searchAll(paraSubList,gap=1e-5)
            for j in range(len(time)):
                allTime.append(time[j])
                allAttri.append(attri[j])
                bestgap.append(gap[j])
                basetime.append(bl[j]['SolvingTime'])
                basegap.append(bl[j]['BestGap'])
        prob=[]
        for i in range(len(proLis)):
            for j in range(16):
                prob.append(proLis[i])
        return allTime,allAttri,bestgap,basetime,basegap,prob

    
    
    def findResult(self,paraSubList=None,maxTime=100000,minTime=0,report_path=None,
                   file_path=os.path.join('miplib-perm-200x2','presolve'),filename='presolve.csv'):
        """ 
        function:用来找到所有参数设置下所有问题的结果属性
        Args:   paramSubList:参数列表 
                maxTime:最大求解时间
                minTime:最小求解时间
                report_path:报告路径
                file_path:文件路径
                filename:保存文件名
        """
        if paraSubList is None:
            paraSubList=self.CutHighLevelSelect
        allTime,allAttri,bestgap,basetime,basegap,nameList=self.findAllPro(maxTime=maxTime,minTime=minTime,
                                                                           paraSubList=paraSubList,
                                                                           report_path=report_path,file_path=file_path) 
        df=pd.DataFrame([allTime, allAttri, bestgap, basetime,basegap,nameList]) 
        df_t=df.T
        df_t.rename(columns={0:' Time',1:' Attri' , 2:'BestGap', 3:'BaseTime', 4:'BaseGap', 5:' Name' }, inplace =True)
        df_t.to_csv(filename)
        
    def getObjective(self):
        """_summary_ 用来得到目标函数值
        """
        return self.model.getObjective()
    def write(self,path):
        """_summary_ 用来将模型写入文件

        Args:
            path (str): 文件路径
        """
        self.model.write(path)
    
    def getObjective(self):
        return self.model.getObjective()
    # def getConstraints(self):
    #     return self.
    def getVars(self):
        return self.model.getVars()
    
    def getConstrs(self):
        return self.model.getConstrs()
    
    def getA(self):
        return self.model.getA()
    
    def varsDict(self):
        dict={}
        vars=self.getVars()
        for i in len(vars):
            dict[i]=vars[i].getName()
        return vars
    
    def generBiparNormal(self,file_path=None,norm_f=False):
        """_summary_ 0 =; 1 <=; 2 >=
            生成二分图的各组件，返回值用于生成二分图
            若file_path不为None，则读入文件后生成为二分图
        Args:file_path (str): 文件路径(默认为None)
        """
        if file_path is not None:
            self.read(file_path)
        A=sp.csr_matrix(self.getA())
        norm={}
        norm['min_edge']=A.min()
        norm['max_edge']=A.max()
        data = A.data
        indices = A.indices
        indptr = A.indptr
        var=self.getVars()
        constr=self.getConstrs()
        m,n=A.shape
        x_s=[]
        x_t=[]
        edge_attr=[]
        edge_index=[[],[]]
        constrIndex=0
        minLB=1e30
        maxLB=-1e30
        minUB=1e30
        maxUB=-1e30
        minObj=1e30
        maxObj=-1e30
        minb=1e30
        maxb=-1e30
        for i in range(n):
            varType=1 if var[i].getType()=='I' else 0
            varLB=var[i].LB
            varUB=var[i].UB
            hasLB=1
            hasUB=1
            if varLB==-1e30:
                varLB=0
                hasLB=0
            if varUB==1e30:
                varUB=0
                hasUB=0
            obj=var[i].Obj
            if obj<minObj:
                minObj=obj
            if obj>maxObj:
                maxObj=obj
            if varLB<minLB:
                minLB=varLB
            if varLB>maxLB:
                maxLB=varLB
            if varUB<minUB:
                minUB=varUB
            if varUB>maxUB:
                maxUB=varUB
            x_t.append([hasLB,hasUB,varLB,varUB,obj,varType])
        norm['minLB']=minLB
        norm['maxLB']=maxLB
        norm['minUB']=minUB
        norm['maxUB']=maxUB
        norm['minObj']=minObj
        norm['maxObj']=maxObj
        
        for i in range(m):
            flag=0
            start_idx = indptr[i]
            end_idx = indptr[i + 1]
            #col_indices = indices[start_idx:end_idx]
            LB=constr[i].LB
            UB=constr[i].UB
            if LB==UB:
                if LB<minb:
                    minb=LB
                if UB>maxb:
                    maxb=UB
                x_s.append([0.5,LB])
                for j in range(start_idx,end_idx):
                    if data[j]==0:
                        continue
                    edge_attr.append(data[j])
                    edge_index[0].append(constrIndex)
                    edge_index[1].append(indices[j])
            elif LB==-1e30 and UB<1e30:
                if UB>maxb:
                    maxb=UB
                if UB<minb:
                    minb=UB
                x_s.append([1,UB])
                for j in range(start_idx,end_idx):
                    if data[j]==0:
                        continue
                    edge_attr.append(data[j])
                    edge_index[0].append(constrIndex)
                    edge_index[1].append(indices[j])
            elif LB>-1e30 and UB==1e30:
                if LB>maxb:
                    maxb=LB
                if LB<minb:
                    minb=LB
                x_s.append([0,LB])
                for j in range(start_idx,end_idx):
                    if data[j]==0:
                        continue
                    edge_attr.append(data[j])
                    edge_index[0].append(constrIndex)
                    edge_index[1].append(indices[j])
            elif LB>-1e30 and UB<1e30:
                if LB>maxb:
                    maxb=LB
                if UB<minb:
                    minb=UB
                if LB<minb:
                    minb=LB
                if UB>maxb:
                    maxb=UB
                    
                flag=1
                x_s.append([1,UB])
                x_s.append([0,LB])
                for j in range(start_idx,end_idx):
                    if data[j]==0:
                        continue
                    edge_attr.append(data[j])
                    edge_index[0].append(constrIndex)
                    edge_index[1].append(indices[j])
                    edge_attr.append(data[j])
                    edge_index[0].append(constrIndex+1)
                    edge_index[1].append(indices[j])
            constrIndex+=2 if flag else 1  
        norm['minb']=minb
        norm['maxb']=maxb     
        edge_index=torch.tensor(edge_index,dtype=torch.int64)
        x_s=torch.tensor(x_s,dtype=torch.float32)
        x_t=torch.tensor(x_t,dtype=torch.float32)
        edge_attr=torch.tensor(edge_attr,dtype=torch.float32).unsqueeze(-1)
        if norm_f:
            # x_s[:,1]=(x_s[:,1]-norm['minb'])/(norm['maxb']-norm['minb'])
            # x_t[:,2]=(x_t[:,2]-norm['minLB'])/(norm['maxLB']-norm['minLB'])
            # x_t[:,3]=(x_t[:,3]-norm['minUB'])/(norm['maxUB']-norm['minUB'])
            # x_t[:,4]=(x_t[:,4]-norm['minObj'])/(norm['maxObj']-norm['minObj'])
            # edge_attr[:]=(edge_attr-norm['min_edge'])/(norm['max_edge']-norm['min_edge'])
            return edge_index,edge_attr,x_s,x_t,norm
        else:
            return edge_index,edge_attr,x_s,x_t,norm
        
         
    def reformulBipartite(self,file_path=None,norm_f=False):
        """_summary_ 用来进行建模为Ax<=b x \in Z+的形式,直接生成二分图
        """
    
        
        if file_path is not None:
            self.read(file_path)
        A=sp.csc_matrix(self.getA())
        norm={}
        maxA=A.max()
        minA=A.min()
        norm['min_edge']=min(-maxA,minA,0)
        norm['max_edge']=max(maxA,-minA,1)
        data = A.data
        indices = A.indices
        indptr = A.indptr
        var=self.getVars()
        constr=self.getConstrs()
        m,n=A.shape
        x_s=[]
        x_t=[]
        #ct=0
 
        edge_attr=[]
        edge_index=[[],[]]
        nodeIndex=0
        minObj=1e30
        maxObj=-1e30
        minb=1e30
        maxb=-1e30
        cons_lis=[]
        cons_ct=0
        #first step 2m constraints
        for i in range(m):
            cons_lis.append([-1,-1])
            UB=constr[i].UB
            LB=constr[i].LB
            if UB<1e30:
                cons_lis[i][0]=cons_ct
                x_s.append([UB])
                cons_ct=cons_ct+1
            if LB>-1e30:
                cons_lis[i][1]=cons_ct
                x_s.append([-LB])
                cons_ct=cons_ct+1              
                

     
        
        for i in range(n):
            obj=var[i].Obj
            varType=1 if var[i].getType()=='I' else 0
            lb=var[i].LB
            ub=var[i].UB
            start_idx = indptr[i]
            end_idx = indptr[i + 1]
          
            #没有考虑x‘的顺序
            if ub==1e30:
                if lb==-1e30:#上下界都没有
                    x_pos=[obj,varType]
                    x_neg=[-obj,varType]
                    x_t.append(x_pos)
                    x_t.append(x_neg)
                    for j in indices[start_idx:end_idx]:
                        idx0=cons_lis[j][0]
                        idx1=cons_lis[j][1]
                        #pos
                        if idx0!=-1:
                            edge_index[0].append(idx0)
                            edge_index[1].append(nodeIndex)
                            edge_attr.append(A[j,i].item())
                            edge_index[0].append(idx0)
                            edge_index[1].append(nodeIndex+1)
                            edge_attr.append(-A[j,i].item())                            
                        if idx1!=-1:
                            edge_index[0].append(idx1)
                            edge_index[1].append(nodeIndex)
                            edge_attr.append(-A[j,i].item())
                            edge_index[0].append(idx1)
                            edge_index[1].append(nodeIndex+1)
                            edge_attr.append(A[j,i].item())
                    nodeIndex=nodeIndex+2
                    continue
                else:#有下界
                    x_t.append([obj,varType])
                    for j in indices[start_idx:end_idx]:
                        idx0=cons_lis[j][0]
                        idx1=cons_lis[j][1]
                        constr_ub=A[j,i].item()*lb
                        if idx0!=-1:
                            edge_index[0].append(idx0)
                            edge_index[1].append(nodeIndex)
                        
                            edge_attr.append(A[j,i].item())
                            x_s[idx0][0]=x_s[idx0][0]+constr_ub
                        if idx1!=-1:
                            edge_index[0].append(idx1)
                            edge_index[1].append(nodeIndex)
                            edge_attr.append(-A[j,i].item())
                            x_s[idx1][0]=x_s[idx1][0]-constr_ub
                    nodeIndex=nodeIndex+1
                    continue
            else:
                if lb==-1e30:#(只有上界）
                    x_t.append([obj,varType])
                    for j in indices[start_idx:end_idx]:
                        idx0=cons_lis[j][0]
                        idx1=cons_lis[j][1]
                        constr_ub=A[j,i].item()*lb
                        if idx0!=-1:
                            edge_index[0].append(idx0)
                            edge_index[1].append(nodeIndex)
                        
                            edge_attr.append(-A[j,i].item())
                            x_s[idx0][0]=x_s[idx0][0]+constr_ub
                        if idx1!=-1:
                            edge_index[0].append(idx1)
                            edge_index[1].append(nodeIndex)
                            edge_attr.append(A[j,i].item())
                            x_s[idx1][0]=x_s[idx1][0]-constr_ub
                    nodeIndex=nodeIndex+1 
                    continue
                else:#上下界都有
                    #ct=ct+1
                    x_t.append([obj,varType])
                    for j in indices[start_idx:end_idx]:
                        idx0=cons_lis[j][0]
                        idx1=cons_lis[j][1]
                        constr_ub=A[j,i].item()*lb
                        if idx0!=-1:
                            edge_index[0].append(idx0)
                            edge_index[1].append(nodeIndex)
                        
                            edge_attr.append(A[j,i].item())
                            x_s[idx0][0]=x_s[idx0][0]+constr_ub
                        if idx1!=-1:
                            edge_index[0].append(idx1)
                            edge_index[1].append(nodeIndex)
                            edge_attr.append(-A[j,i].item())
                            x_s[idx1][0]=x_s[idx1][0]-constr_ub
                    nodeIndex=nodeIndex+1     
                    x_s.append([ub-lb])
                    edge_index[0].append(len(x_s)-1)
                    edge_index[1].append(nodeIndex)
                    nodeIndex=nodeIndex+1   
        edge_index=torch.tensor(edge_index,dtype=torch.int64)
        x_s=torch.tensor(x_s,dtype=torch.float32)
        x_t=torch.tensor(x_t,dtype=torch.float32)                    
        '''print(len(x_s))
        print(len(x_t))
        print(n)
        print(m)
        print(ct)
        print(cons_ct)'''
        if norm_f:#统计入度
            x_s[:,0]=(x_s[:,0]-norm['minb'])/(norm['maxb']-norm['minb'])

            x_t[:,0]=(x_t[:,0]-norm['minObj'])/(norm['maxObj']-norm['minObj'])
            edge_attr[:]=(edge_attr-norm['min_edge'])/(norm['max_edge']-norm['min_edge'])
            return edge_index,edge_attr,x_s,x_t,norm
        else:
            return edge_index,edge_attr,x_s,x_t,None

    def calculate(self,my_array):
        if not my_array:
            mean_value, variance_value, median_value = 0, 0, 0
        else:
            mean_value = np.mean(my_array)
            variance_value = np.var(my_array)
            median_value = np.median(my_array)
        return mean_value,variance_value,median_value

    def generStatic(self,file_path=None,norm_f=False):
        if file_path is not None:
            self.read(file_path)
        A=sp.csr_matrix(self.getA())
        norm={}
       # norm['min_edge']=A.min()
        #norm['max_edge']=A.max()
        data = A.data
        indices = A.indices
        indptr = A.indptr
        var=self.getVars()
        constr=self.getConstrs()
        m,n=A.shape
        x_s=[]
        x_t=[]
        edge_attr=[]
        ratio=[]
        edge_index=[[],[]]
        stat_Var=[]
        stat_Con=[]
        constrIndex=0
        minLB=1e30
        maxLB=-1e30
        minUB=1e30
        maxUB=-1e30
        minObj=1e30
        maxObj=-1e30
        minb=1e30
        maxb=-1e30
        
        obj_b=[]
        for i in range(n):
            if var[i].getType()=='I':
                varType=1
            elif var[i].getType()=='C':
                varType=2
            elif var[i].getType()=='B':
                varType=3
            else:
                varType=0
            varLB=var[i].LB
            varUB=var[i].UB
            hasLB=1
            hasUB=1
            if varLB==-1e30:
                varLB=0
                hasLB=0
            if varUB==1e30:
                varUB=0
                hasUB=0
            obj=var[i].Obj
            if obj<minObj and obj!=0:
                minObj=obj
            if obj>maxObj and obj!=0:
                maxObj=obj
            if varLB<minLB:
                minLB=varLB
            if varLB>maxLB:
                maxLB=varLB
            if varUB<minUB:
                minUB=varUB
            if varUB>maxUB:
                maxUB=varUB
            x_t.append([hasLB,hasUB,varLB,varUB,obj,varType])
        col=['hasLB','hasUB','varLB','varUB','obj','varType']
        xt=pd.DataFrame(x_t,columns=col)
        norm['Rows']=math.log(m)
        norm['Columns']=math.log(n)
        norm['Nonzeros']=np.count_nonzero(data)/(m*n)
        # norm['Symmetries']=np.allclose(A.toarray(), A.T.toarray())
        
        norm['per_i']=xt['varType'].value_counts().get(1, 0)/n
        #norm['per_c']=xt['varType'].value_counts().get(2, 0)/n
        norm['per_b']=xt['varType'].value_counts().get(3, 0)/n
        #norm['per_others']=xt['varType'].value_counts().get(0, 0)/n
        
        # if minObj!=1e30:
        #     norm['minObj']=minObj
        # else:
        #     norm['minObj']=0
        # if maxObj!=-1e30:
        #     norm['maxObj']=maxObj
        # else:
        #     norm['maxObj']=0
        '''
        norm['mean_obj']=xt['obj'].mean()
        norm['dev_obj']=xt['obj'].var()
        norm['med_obj']=xt['obj'].median()
        norm['has_obj']=1-xt['obj'].value_counts().get(0, 0)/n
        '''
        norm['obj_dynamic']=math.log(abs(maxObj/minObj)+1)
        norm['has_varlb']=xt['hasLB'].value_counts().get(1, 0)/n
        norm['has_varub']=xt['hasUB'].value_counts().get(1, 0)/n
        '''
        i=xt[xt['varType']==1]
        c=xt[xt['varType']==2]
        b=xt[xt['varType']==3]
        
        if max(0,i.shape[0])==0:
            n1=n
        else:
            n1=i.shape[0]
        if max(0,c.shape[0])==0:
            n2=n
        else:
            n2=c.shape[0]
        if max(0,b.shape[0])==0:
            n3=n
        else:
            n3=b.shape[0]
        
        norm['mean_obj_i']=i['obj'].mean()
        norm['dev_obj_i']=i['obj'].var()
        norm['med_obj_i']=i['obj'].median()
        norm['has_obj_i']=i[i['obj']!=0]['obj'].count()/n1
        
        norm['mean_obj_c']=c['obj'].mean()
        norm['dev_obj_c']=c['obj'].var()
        norm['med_obj_c']=c['obj'].median()
        norm['has_obj_c']=c[c['obj']!=0]['obj'].count()/n2
        
        norm['mean_obj_b']=b['obj'].mean()
        norm['dev_obj_b']=b['obj'].var()
        norm['med_obj_b']=b['obj'].median()
        norm['has_obj_b']=b[b['obj']!=0]['obj'].count()/n3
             
        norm['min_varlb']=minLB
        norm['max_varlb']=maxLB
        norm['mean_varlb']=xt[xt['hasLB'] == 1]['varLB'].mean()
        norm['dev_varlb']=xt[xt['hasLB'] == 1]['varLB'].var()
        norm['med_varlb']=xt[xt['hasLB'] == 1]['varLB'].median()
        norm['has_varlb']=xt['hasLB'].value_counts().get(1, 0)/n
        
        norm['min_varub']=minUB
        norm['max_varub']=maxUB
        norm['mean_varub']=xt[xt['hasUB'] == 1]['varUB'].mean()
        norm['dev_varub']=xt[xt['hasUB'] == 1]['varUB'].var()
        norm['med_varub']=xt[xt['hasUB'] == 1]['varUB'].median()
        norm['has_varub']=xt['hasUB'].value_counts().get(1, 0)/n
        
        
        norm['min_varlb_i']=i['varLB'].min()
        norm['max_varlb_i']=i[i['varLB']!=1e30]['varLB'].max()
        norm['mean_varlb_i']=i[i['hasLB'] == 1]['varLB'].mean()
        norm['dev_varlb_i']=i[i['hasLB'] == 1]['varLB'].var()
        norm['med_varlb_i']=i[i['hasLB'] == 1]['varLB'].median()
        norm['has_varlb_i']=i['hasLB'].value_counts().get(1, 0)/n1
        norm['min_varub_i']=i[i['varUB']!=-1e30]['varUB'].min()
        norm['max_varub_i']=i['varUB'].max()
        norm['mean_varub_i']=i[i['hasUB'] == 1]['varUB'].mean()
        norm['dev_varub_i']=i[i['hasUB'] == 1]['varUB'].var()
        norm['med_varub_i']=i[i['hasUB'] == 1]['varUB'].median()
        norm['has_varub_i']=i['hasUB'].value_counts().get(1, 0)/n1
        
        
        norm['min_varlb_c']=c['varLB'].min()
        norm['max_varlb_c']=c[c['varLB']!=1e30]['varLB'].max()
        norm['mean_varlb_c']=c[c['hasLB'] == 1]['varLB'].mean()
        norm['dev_varlb_c']=c[c['hasLB'] == 1]['varLB'].var()
        norm['med_varlb_c']=c[c['hasLB'] == 1]['varLB'].median()
        norm['has_varlb_c']=c['hasLB'].value_counts().get(1, 0)/n2
        norm['min_varub_c']=c[c['varUB']!=-1e30]['varUB'].min()
        norm['max_varub_c']=c['varUB'].max()
        norm['mean_varub_c']=c[c['hasUB'] == 1]['varUB'].mean()
        norm['dev_varub_c']=c[c['hasUB'] == 1]['varUB'].var()
        norm['med_varub_c']=c[c['hasUB'] == 1]['varUB'].median()
        norm['has_varub_c']=c['hasUB'].value_counts().get(1, 0)/n2
        
        
        norm['min_varlb_b']=b['varLB'].min()
        norm['max_varlb_b']=b[b['varLB']!=1e30]['varLB'].max()
        norm['mean_varlb_b']=b[b['hasLB'] == 1]['varLB'].mean()
        norm['dev_varlb_b']=b[b['hasLB'] == 1]['varLB'].var()
        norm['med_varlb_b']=b[b['hasLB'] == 1]['varLB'].median()
        norm['has_varlb_b']=b['hasLB'].value_counts().get(1, 0)/n3
        norm['min_varub_b']=b[b['varUB']!=-1e30]['varUB'].min()
        norm['max_varub_b']=b['varUB'].max()
        norm['mean_varub_b']=b[b['hasUB'] == 1]['varUB'].mean()
        norm['dev_varub_b']=b[b['hasUB'] == 1]['varUB'].var()
        norm['med_varub_b']=b[b['hasUB'] == 1]['varUB'].median()
        norm['has_varub_b']=b['hasUB'].value_counts().get(1, 0)/n3
        
        '''
        
        
        AGG=0
        VBD=0
        PAR=0
        PAC=0
        COV=0
        CAR=0
        EQK=0
        BIN=0
        IVK=0
        KNA=0
        IKN=0
        M01=0
        MI=0
        GEN=0
        con_i=[]
        con_c=[]
        con_b=[]
        den=[]
        clique=[]
        
        for i in range(m):
            start_idx = indptr[i]
            end_idx = indptr[i + 1]
            # amin=float('-inf')
            # amin2=float('-inf')
            negsum=0
            cmin=0
            nbin=end_idx-start_idx
            flag=0
            #col_indices = indices[start_idx:end_idx]
            LB=constr[i].LB
            UB=constr[i].UB
            sub=data[start_idx:end_idx]
            sub=[x for x in sub if x != 0]
            sorted_sub = sorted([abs(x) for x in sub])
            # amin=sorted_sub[0]
            # amin2=sorted_sub[1]
            ratio.append(max(sub)/min(sub))
            cnt=0
            cnt1=0
            cnt2=0
            f=1
            f1=1
            f2=0
            equal_lb=0
            equal_ub=0
            for j in range(start_idx,end_idx):
                edge_attr.append(data[j])
                if var[indices[j]].getType()=='B':
                    cnt+=1
                    con_b.append(data[j])
                if var[indices[j]].getType()=='I':
                    cnt1+=1
                    con_i.append(data[j])
                if var[indices[j]].getType()=='C':
                    con_c.append(data[j]) 
                if data[j]==LB:
                    equal_lb=1
                if data[j]==UB:
                    equal_ub=1   
                if data[j]!=1:
                    f=0
                if data[j]-int(data[j])!=0:
                    f1=0
                if data[j]<0:
                    negsum+=data[j]
            if end_idx-start_idx==cnt:
                f2=1
            den.append((end_idx-start_idx)/n)
            #print(UB,cmin,negsum,amin,amin2,nbin)
            # if UB-cmin-negsum-amin<amin2 and nbin>0:
            #     clique.append(nbin/n)
            if LB==UB:
                if LB<minb:
                    minb=LB
                if UB>maxb:
                    maxb=UB
                x_s.append([LB,UB,1,1,1])
                #if end_idx-start_idx==2:
                 #   AGG+=1
                if f and f2 and LB==1:
                    PAR+=1
                elif f and f2 and LB-int(LB)==0:
                    CAR+=1  
                elif f2 and f1 and LB-int(LB)==0:
                    EQK+=1
                elif cnt!=0 and cnt1==0:
                    M01+=1
                elif cnt1!=0 and cnt==0:
                    MI+=1
                else:
                    GEN+=1
            elif LB==-1e30 and UB<1e30:
                if UB>maxb:
                    maxb=UB
                if UB<minb:
                    minb=UB
                x_s.append([LB,UB,0,1,2])
                #if end_idx-start_idx==2 and (data[start_idx]==1 or data[end_idx-1]==1) and var[indices[start_idx]].getType()!='B' and var[indices[end_idx-1]].getType()!='B':
               #     VBD+=1
                if f and f2 and UB==1:
                    PAC+=1
                elif f and f2 and UB-int(UB)==0:
                    IVK+=1
                elif f2 and f1 and equal_ub:
                    BIN+=1
                elif f2 and f1:
                    KNA+=1
                elif end_idx-start_idx==cnt1 and f1:
                    IKN+=1
                elif cnt!=0 and cnt1==0:
                    M01+=1
                elif cnt1!=0 and cnt==0:
                    MI+=1
                else:
                    GEN+=1
                
            elif LB>-1e30 and UB==1e30:
                if LB>maxb:
                    maxb=LB
                if LB<minb:
                    minb=LB
                x_s.append([LB,UB,1,0,3])
                #if end_idx-start_idx==2 and (data[start_idx]==1 or data[end_idx-1]==1) and var[indices[start_idx]].getType()!='B' and var[indices[end_idx-1]].getType()!='B':
                 #   VBD+=1
                if f and f2 and LB==1:
                    COV+=1
                elif f2 and f1 and equal_lb:
                    BIN+=1
                elif f2 and f1:
                    KNA+=1
                elif end_idx-start_idx==cnt1 and f1:
                    IKN+=1
                elif cnt!=0:
                    M01+=1
                elif cnt1!=0 and cnt==0:
                    MI+=1
                else:
                    GEN+=1
               
            elif LB>-1e30 and UB<1e30:
                if LB>maxb:
                    maxb=LB
                if UB<minb:
                    minb=UB
                if LB<minb:
                    minb=LB
                if UB>maxb:
                    maxb=UB
                GEN+=1 
                flag=1
                x_s.append([LB,UB,1,1,4])
                
            constrIndex+=2 if flag else 1 
        
        col=['LB','UB','hasLB','hasUB','Type']
        xs=pd.DataFrame(x_s,columns=col)
        #norm['con_cnt']=math.log(m)
        norm['Equality']=xs['Type'].value_counts().get(1, 0)/m
        norm['GreaterThan']=xs['Type'].value_counts().get(2, 0)/m
        norm['LessThan']=xs['Type'].value_counts().get(3, 0)/m
        norm['RHS_dynamic']=math.log(abs(maxb/(abs(minb)+1))+1)
        norm['Coe_dynamic']=math.log(abs(max(data)/min(data)))
        '''
        norm['density']=np.count_nonzero(data)/(m*n)
        norm['den_max']=max(den)
        norm['den_min']=min(den)
        norm['den_mean']=np.mean(den)
        norm['den_var']=np.var(den)
        norm['den_median']=np.median(den)
        
        norm['clique_cnt']=math.log(len(clique)) if clique !=[] else 0
        norm['clique_max']=max(clique) if clique !=[] else 0
        norm['clique_min']=min(clique) if clique !=[] else 0
        norm['clique_mean']=np.mean(clique) if clique !=[] else 0
        norm['clique_var']=np.var(clique) if clique !=[] else 0
        norm['clique_median']=np.median(clique) if clique !=[] else 0

        
        norm['mean_conlb']=xs[xs['hasLB'] == 1]['LB'].mean()
        norm['dev_conlb']=xs[xs['hasLB'] == 1]['LB'].var()
        norm['med_conlb']=xs[xs['hasLB'] == 1]['LB'].median()
        norm['has_conlb']=xs['hasLB'].value_counts().get(1, 0)/m
        norm['lb_density']=1-xs['LB'].value_counts().get(-1e30, 0)/m
        
        norm['mean_conub']=xs[xs['hasUB'] == 1]['UB'].mean()
        norm['dev_conub']=xs[xs['hasUB'] == 1]['UB'].var()
        norm['med_conub']=xs[xs['hasUB'] == 1]['UB'].median()
        norm['has_conub']=xs['hasUB'].value_counts().get(1, 0)/m
        norm['ub_density']=1-xs['UB'].value_counts().get(1e30, 0)/m
        
        norm['minb']=minb
        norm['maxb']=maxb  
        
        norm['max_coe']=max(edge_attr)
        norm['min_coe']=min(edge_attr)
        norm['mean_coe']=np.mean(edge_attr)
        norm['var_coe']=np.var(edge_attr)
        norm['med_coe']=np.median(edge_attr)
        
        norm['max_coe_i']=max(con_i,default=0)
        norm['min_coe_i']=min(con_i,default=0)
        norm['mean_coe_i'],norm['var_coe_i'],norm['med_coe_i']=self.calculate(con_i)
        
        norm['max_coe_c']=max(con_c,default=0)
        norm['min_coe_c']=min(con_c,default=0)
        norm['mean_coe_c'],norm['var_coe_c'],norm['med_coe_c']=self.calculate(con_c)
        
        norm['max_coe_b']=max(con_b,default=0)
        norm['min_coe_b']=min(con_b,default=0)
        norm['mean_coe_b'],norm['var_coe_b'],norm['med_coe_b']=self.calculate(con_b)
        
        norm['max_ratio']=max(ratio)
        norm['min_ratio']=min(ratio)
        norm['mean_ratio']=np.mean(ratio)
        norm['var_ratio']=np.var(ratio)
        norm['med_ratio']=np.median(ratio)
        '''
        
        #norm['AGG']=AGG/m
        #norm['VBD']=VBD/m
        norm['PAR']=PAR/m #set partition
        norm['PAC']=PAC/m #set packing
        norm['COV']=COV/m #set cover
        norm['CAR']=CAR/m #cardinality
        norm['EQK']=EQK/m #equality knapsack
        norm['BIN']=(BIN)/m #bin packing
        #norm['IVK']=IVK/m #
        norm['KNA']=(KNA+IVK)/m #knapsack
        norm['IKN']=IKN/m #integer knapsack
        norm['M01']=M01/m #mixed binary
        norm['MI']=MI/m #mixed integer
        norm['CON']=GEN/m #continupus
        
        
        norm = pd.DataFrame(norm,index=[0])
        return edge_index,edge_attr,x_s,x_t,norm
