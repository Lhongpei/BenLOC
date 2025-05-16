import os
from itertools import product

import coptpy as cp
import numpy as np
import pandas as pd
import torch
from coptpy import COPT
from scipy import sparse as sp
from tqdm import tqdm


class MIPmodel:
    def __init__(self, name='default', file_path=None, log=False):
        self.env = cp.Envr()
        self.model = self.env.createModel(name)
        self.baseline = None
        self.hasRead = False
        self.log = log
        self.resultLis = ["SolvingTime", "Nodecnt", "BestGap"]
        self.paraLis = ['CutLevel', 'RootCutLevel', 'TreeCutLevel', 'RootCutRounds', 'NodeCutRounds', 'HeurLevel',
                        'RoundingHeurLevel', 'DivingHeurLevel',
                        'SubMipHeurLevel', 'StrongBranching', 'ConflictAnalysis', 'MipStartMode', 'MipStartNodeLimit']
        self.paraRange = {
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
        self.HignLevelSelect = ["RootCutLevel", "TreeCutLevel", "RoundingHeurLevel", "DivingHeurLevel",
                                "SubMipHeurLevel", "StrongBranching"]
        self.CutHighLevelSelect = ["RootCutLevel", "TreeCutLevel"]
        self.HeurHighLevelSelect = ["RoundingHeurLevel", "DivingHeurLevel", "SubMipHeurLevel", "StrongBranching"]
        if self.log:
            self.model.setParam('Logging', 1)
        else:
            self.model.setParam('Logging', 0)
        if not file_path is None:
            self.read(file_path)

    def read(self, file_path):
        """
        用于根据文件路径读入模型
    
        参数:
        file_path:文件路径
        """
        self.model.read(file_path)
        self.baseline = None
        self.hasRead = True

    def solve(self):
        """_summary_ 用来求解MIP问题
        """
        if self.hasRead:
            self.model.solve()
        else:
            raise ResourceWarning('Hasn\'t define MIP problem')

    def resetParam(self):
        self.model.resetParam()
        if self.log == False:
            self.model.setParam('Logging', 0)

    def runbaseline(self):
        """_summary_ 用来求解默认参数设置下的结果属性
        """
        self.resetParam()
        self.solve()
        self.baseline = self.showResult()
        for i in self.resultLis:
            self.baseline[i] = self.model.getAttr(i)

    def getBaseline(self):
        """
        得到在默认参数设置下的结果属性
        """
        if self.baseline is None:
            self.runbaseline()
        return self.baseline

    def setParamByDict(self, paraDict):
        for i in paraDict:
            self.model.setParam(i, paraDict[i])

    def showResult(self):
        """_summary_ 用来显示结果属性

        Returns:
            dict: 返回一个字典，键为结果属性名称，值为结果属性值
        """
        self.solve()
        tmp = {}
        for i in self.resultLis:
            tmp[i] = self.model.getAttr(i)
        return tmp

    def testParam(self, paraDict):
        """_summary_ 用来测试参数设置下的结果属性

        Args:
            paraDict (字典): 该字典的键为参数名，值为参数值

        Returns:
            tuple: 返回元组，第一个元素为求解时间，第二个元素为节点数，第三个元素为最优解的gap.
        """
        self.setParamByDict(paraDict)
        result = self.showResult()
        return result

    def setTimeLimit(self, time):
        """_summary_ 用来设置求解时间上限

        Args:
            time (int): 求解时间上限
        """
        self.model.setParam('TimeLimit', time)

    def getParamName(self):
        """_summary_ 用来得到所有参数的名称

        Returns:
            list: 返回所有参数的名称
        """
        return self.paraLis

    def setParamByLis(self, setLis):
        """_summary_ 用列表来设置参数

        Args:
            paraLis (list): 参数值的列表
        """
        assert len(setLis) == len(self.paraLis)
        dict = zip(self.paraLis, setLis)
        self.setParamByDict(dict)

    def getParam(self):
        """_summary_ 用来得到所有参数的值

        Returns:
            dict: 返回一个字典，键为参数名，值为参数值
        """
        tmp = {}
        for i in self.paraLis:
            tmp[i] = self.model.getAttr(i)
        return tmp

    def bruteFindAll(self, paraSubList, timelimit=10000, accelerate=False):
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
        resultForParams = {}
        for params in parameters_list:
            params_tuple = tuple(params.items())  # Convert the dictionary to a tuple of key-value pairs
            self.setTimeLimit(timelimit)
            result = self.testParam(params)
            # 判断是否加速
            if accelerate and result['SolvingTime'] < timelimit and result['BestGap'] < 1e-8:
                timelimit = result['SolvingTime']
                self.setTimeLimit(timelimit)

            resultForParams[params_tuple] = result
        return resultForParams

    def bruteMinTime(self, paraSubList, timelimit=10000, accelerate=False):
        """_summary_ 用来暴力搜索参数设置下的最小求解时间

        Args:
            paraSubList (list): 选择属性的列表
        Returns:
            tuple: 返回一个元组，第一个元素为最小求解时间，第二个元素为参数设置
        """
        baseline = self.getBaseline()
        timelimit = baseline['SolvingTime'] + 0.5
        print('Baseline:', baseline)
        result = self.bruteFindAll(paraSubList, accelerate=accelerate, timelimit=timelimit)
        minTime = 100000
        minParams = None
        for i in result:
            if result[i]['SolvingTime'] < minTime and result[i]['BestGap'] < 1e-8:
                minTime = result[i]['SolvingTime']
                minParams = i
        return minTime, minParams

    def searchAll(self, paraSubList, gap, timelimit=1000):
        """_summary_ 搜索所有参数设置下的结果属性
        
        return: 列表，第一个元素为求解时间，第二个元素为参数,第三个元素为最优解的gap，第四个元素为baseline
        """
        baseline = self.getBaseline()
        subset_parameters = {param: self.paraRange[param] for param in paraSubList}

        parameter_combinations = list(product(*subset_parameters.values()))
        parameters_list = []
        for combination in parameter_combinations:
            parameters_dict = {key: value for key, value in zip(subset_parameters.keys(), combination)}
            parameters_list.append(parameters_dict)
        resultForParams = {}
        for params in parameters_list:
            params_tuple = tuple(params.items())  # Convert the dictionary to a tuple of key-value pairs
            self.setTimeLimit(timelimit)
            result = self.testParam(params)

            # if result['SolvingTime']<timelimit and result['BestGap']< gap:
            #   timelimit=result['SolvingTime']
            #   self.setTimeLimit(timelimit)

            resultForParams[params_tuple] = result
            index = []
            solvingtime = []
            basel = []
            bestgap = []
            for i in resultForParams:
                index.append(i)
                solvingtime.append(resultForParams[i]['SolvingTime'])
                basel.append(baseline)
                bestgap.append(resultForParams[i]['BestGap'])
        return solvingtime, index, bestgap, basel

    def findAllPro(self, maxTime, minTime, paraSubList, file_path, report_path=None):
        """_summary_ 用来找到所有参数设置下的结果属性
        """
        index_set = []
        if not report_path is None:
            allreport = os.listdir(report_path)
            choReport_path = os.path.join(report_path, allreport[0])
            df = pd.read_csv(choReport_path)
            for index, i in enumerate(df['Time']):
                if i < maxTime and i > minTime:
                    index_set.append(index)
        allpro = os.listdir(file_path)

        # proLis=[allpro[self.findFileName(file_path,df['Name'][i]+'.mps.gz')]for i in index_set]
        proLis = allpro
        allTime = []
        allAttri = []
        bestgap = []
        basetime = []
        basegap = []
        for i in tqdm(proLis, desc='Solving process:'):
            print('Solving: ' + i)
            choReport_path = os.path.join(file_path, i)
            self.read(choReport_path)
            time, attri, gap, bl = self.searchAll(paraSubList, gap=1e-5)
            for j in range(len(time)):
                allTime.append(time[j])
                allAttri.append(attri[j])
                bestgap.append(gap[j])
                basetime.append(bl[j]['SolvingTime'])
                basegap.append(bl[j]['BestGap'])
        prob = []
        for i in range(len(proLis)):
            for j in range(16):
                prob.append(proLis[i])
        return allTime, allAttri, bestgap, basetime, basegap, prob

    def findResult(self, paraSubList=None, maxTime=100000, minTime=0, report_path=None,
                   file_path=os.path.join('miplib-perm-200x2', 'presolve'), filename='presolve.csv'):
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
            paraSubList = self.CutHighLevelSelect
        allTime, allAttri, bestgap, basetime, basegap, nameList = self.findAllPro(maxTime=maxTime, minTime=minTime,
                                                                                  paraSubList=paraSubList,
                                                                                  report_path=report_path,
                                                                                  file_path=file_path)
        df = pd.DataFrame([allTime, allAttri, bestgap, basetime, basegap, nameList])
        df_t = df.T
        df_t.rename(columns={0: ' Time', 1: ' Attri', 2: 'BestGap', 3: 'BaseTime', 4: 'BaseGap', 5: ' Name'},
                    inplace=True)
        df_t.to_csv(filename)

    def getObjective(self):
        """_summary_ 用来得到目标函数值
        """
        return self.model.getObjective()

    def write(self, path):
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
        dict = {}
        vars = self.getVars()
        for i in len(vars):
            dict[i] = vars[i].getName()
        return vars

    def generBiparNormal(self, file_path=None, norm_f=False):
        """_summary_ 0 =; 1 <=; 2 >=
            生成二分图的各组件，返回值用于生成二分图
            若file_path不为None，则读入文件后生成为二分图
        Args:file_path (str): 文件路径(默认为None)
        """
        try:
            if file_path is not None:
                self.read(file_path)
            A = sp.csr_matrix(self.getA())
            norm = {}
            norm['min_edge'] = A.min()
            norm['max_edge'] = A.max()
            data = A.data
            indices = A.indices
            indptr = A.indptr
            var = self.getVars()
            constr = self.getConstrs()
            m, n = A.shape
            x_s = []
            x_t = []
            edge_attr = []
            edge_index = [[], []]
            constrIndex = 0
            for i in range(n):
                varType = 1 if var[i].getType() in [COPT.BINARY, COPT.INTEGER] else 0
                varLB = var[i].LB
                varUB = var[i].UB
                hasLB = 1
                hasUB = 1
                if varLB == -COPT.INFINITY:
                    varLB = 0
                    hasLB = 0
                if varUB == COPT.INFINITY:
                    varUB = 0
                    hasUB = 0
                obj = var[i].Obj

                assert -COPT.INFINITY < varLB < COPT.INFINITY
                assert -COPT.INFINITY < varUB < COPT.INFINITY
                assert -COPT.INFINITY < obj < COPT.INFINITY
                x_t.append([1, hasLB, hasUB, varLB, varUB, obj, varType])

            for i in range(m):
                start_idx = indptr[i]
                end_idx = indptr[i + 1]
                # col_indices = indices[start_idx:end_idx]
                LB = constr[i].LB
                UB = constr[i].UB
                if LB == -COPT.INFINITY and UB < COPT.INFINITY:
                    x_s.append([0, 0, 0, 1, UB])
                    for j in range(start_idx, end_idx):
                        if data[j] == 0:
                            continue
                        edge_attr.append(data[j])
                        edge_index[0].append(constrIndex)
                        edge_index[1].append(indices[j])
                elif LB > -COPT.INFINITY and UB == COPT.INFINITY:
                    x_s.append([0, 1, LB, 0, 0])
                    for j in range(start_idx, end_idx):
                        if data[j] == 0:
                            continue
                        edge_attr.append(data[j])
                        edge_index[0].append(constrIndex)
                        edge_index[1].append(indices[j])
                elif LB > -COPT.INFINITY and UB < COPT.INFINITY:
                    x_s.append([0, 1, LB, 1, UB])
                    for j in range(start_idx, end_idx):
                        if data[j] == 0:
                            continue
                        edge_attr.append(data[j])
                        edge_index[0].append(constrIndex)
                        edge_index[1].append(indices[j])
                else:
                    raise ValueError('Wrong constr')
                constrIndex += 1
            edge_index = torch.tensor(edge_index, dtype=torch.int64)
            x_s = torch.tensor(x_s, dtype=torch.float32)
            x_t = torch.tensor(x_t, dtype=torch.float32)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)

            norm['minLB_s'] = x_s[:, 2].min()
            norm['maxLB_s'] = x_s[:, 2].max()
            norm['minUB_s'] = x_s[:, 4].min()
            norm['maxUB_s'] = x_s[:, 4].max()
            norm['minLB_t'] = x_t[:, 3].min()
            norm['maxLB_t'] = x_t[:, 3].max()
            norm['minUB_t'] = x_t[:, 4].min()
            norm['maxUB_t'] = x_t[:, 4].max()
            norm['minObj'] = x_t[:, 5].min()
            norm['maxObj'] = x_t[:, 5].max()

            if norm_f:
                # x_s[:,1]=(x_s[:,1]-norm['minb'])/(norm['maxb']-norm['minb'])
                # x_t[:,2]=(x_t[:,2]-norm['minLB'])/(norm['maxLB']-norm['minLB'])
                # x_t[:,3]=(x_t[:,3]-norm['minUB'])/(norm['maxUB']-norm['minUB'])
                # x_t[:,4]=(x_t[:,4]-norm['minObj'])/(norm['maxObj']-norm['minObj'])
                # edge_attr[:]=(edge_attr-norm['min_edge'])/(norm['max_edge']-norm['min_edge'])
                return edge_index, edge_attr, x_s, x_t, norm
            else:
                return edge_index, edge_attr, x_s, x_t, norm
        except Exception as e:
            print(file_path)
            raise e
        
    def reformulBipartite(self, file_path=None, norm_f=False):
        """_summary_ 用来进行建模为Ax<=b x \in Z+的形式,直接生成二分图
        """

        if file_path is not None:
            self.read(file_path)
        A = sp.csc_matrix(self.getA())
        norm = {}
        maxA = A.max()
        minA = A.min()
        norm['min_edge'] = min(-maxA, minA, 0)
        norm['max_edge'] = max(maxA, -minA, 1)
        data = A.data
        indices = A.indices
        indptr = A.indptr
        var = self.getVars()
        constr = self.getConstrs()
        m, n = A.shape
        x_s = []
        x_t = []
        # ct=0

        edge_attr = []
        edge_index = [[], []]
        nodeIndex = 0
        cons_lis = []
        cons_ct = 0
        # first step 2m constraints
        for i in range(m):
            cons_lis.append([-1, -1])
            UB = constr[i].UB
            LB = constr[i].LB
            if UB < COPT.INFINITY:
                cons_lis[i][0] = cons_ct
                x_s.append([UB])
                cons_ct = cons_ct + 1
            if LB > -COPT.INFINITY:
                cons_lis[i][1] = cons_ct
                x_s.append([-LB])
                cons_ct = cons_ct + 1

        for i in range(n):
            obj = var[i].Obj
            varType = 1 if var[i].getType() == 'I' else 0
            lb = var[i].LB
            ub = var[i].UB
            start_idx = indptr[i]
            end_idx = indptr[i + 1]

            # 没有考虑x‘的顺序
            if ub == COPT.INFINITY:
                if lb == -COPT.INFINITY:  # 上下界都没有
                    x_pos = [obj, varType]
                    x_neg = [-obj, varType]
                    x_t.append(x_pos)
                    x_t.append(x_neg)
                    for j in indices[start_idx:end_idx]:
                        idx0 = cons_lis[j][0]
                        idx1 = cons_lis[j][1]
                        # pos
                        if idx0 != -1:
                            edge_index[0].append(idx0)
                            edge_index[1].append(nodeIndex)
                            edge_attr.append(A[j, i].item())
                            edge_index[0].append(idx0)
                            edge_index[1].append(nodeIndex + 1)
                            edge_attr.append(-A[j, i].item())
                        if idx1 != -1:
                            edge_index[0].append(idx1)
                            edge_index[1].append(nodeIndex)
                            edge_attr.append(-A[j, i].item())
                            edge_index[0].append(idx1)
                            edge_index[1].append(nodeIndex + 1)
                            edge_attr.append(A[j, i].item())
                    nodeIndex = nodeIndex + 2
                    continue
                else:  # 有下界
                    x_t.append([obj, varType])
                    for j in indices[start_idx:end_idx]:
                        idx0 = cons_lis[j][0]
                        idx1 = cons_lis[j][1]
                        constr_ub = A[j, i].item() * lb
                        if idx0 != -1:
                            edge_index[0].append(idx0)
                            edge_index[1].append(nodeIndex)

                            edge_attr.append(A[j, i].item())
                            x_s[idx0][0] = x_s[idx0][0] + constr_ub
                        if idx1 != -1:
                            edge_index[0].append(idx1)
                            edge_index[1].append(nodeIndex)
                            edge_attr.append(-A[j, i].item())
                            x_s[idx1][0] = x_s[idx1][0] - constr_ub
                    nodeIndex = nodeIndex + 1
                    continue
            else:
                if lb == -COPT.INFINITY:  # (只有上界）
                    x_t.append([obj, varType])
                    for j in indices[start_idx:end_idx]:
                        idx0 = cons_lis[j][0]
                        idx1 = cons_lis[j][1]
                        constr_ub = A[j, i].item() * lb
                        if idx0 != -1:
                            edge_index[0].append(idx0)
                            edge_index[1].append(nodeIndex)

                            edge_attr.append(-A[j, i].item())
                            x_s[idx0][0] = x_s[idx0][0] + constr_ub
                        if idx1 != -1:
                            edge_index[0].append(idx1)
                            edge_index[1].append(nodeIndex)
                            edge_attr.append(A[j, i].item())
                            x_s[idx1][0] = x_s[idx1][0] - constr_ub
                    nodeIndex = nodeIndex + 1
                    continue
                else:  # 上下界都有
                    # ct=ct+1
                    x_t.append([obj, varType])
                    for j in indices[start_idx:end_idx]:
                        idx0 = cons_lis[j][0]
                        idx1 = cons_lis[j][1]
                        constr_ub = A[j, i].item() * lb
                        if idx0 != -1:
                            edge_index[0].append(idx0)
                            edge_index[1].append(nodeIndex)

                            edge_attr.append(A[j, i].item())
                            x_s[idx0][0] = x_s[idx0][0] + constr_ub
                        if idx1 != -1:
                            edge_index[0].append(idx1)
                            edge_index[1].append(nodeIndex)
                            edge_attr.append(-A[j, i].item())
                            x_s[idx1][0] = x_s[idx1][0] - constr_ub
                    nodeIndex = nodeIndex + 1
                    x_s.append([ub - lb])
                    edge_index[0].append(len(x_s) - 1)
                    edge_index[1].append(nodeIndex)
                    nodeIndex = nodeIndex + 1
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        x_s = torch.tensor(x_s, dtype=torch.float32)
        x_t = torch.tensor(x_t, dtype=torch.float32)
        '''print(len(x_s))
        print(len(x_t))
        print(n)
        print(m)
        print(ct)
        print(cons_ct)'''
        if norm_f:  # 统计入度
            x_s[:, 0] = (x_s[:, 0] - norm['minb']) / (norm['maxb'] - norm['minb'])

            x_t[:, 0] = (x_t[:, 0] - norm['minObj']) / (norm['maxObj'] - norm['minObj'])
            edge_attr[:] = (edge_attr - norm['min_edge']) / (norm['max_edge'] - norm['min_edge'])
            return edge_index, edge_attr, x_s, x_t, norm
        else:
            return edge_index, edge_attr, x_s, x_t, None

    def generBiparN2N(self, file_path=None):
        """ 对每个节点分别做Normalization
            生成二分图的各组件，返回值用于生成二分图
            若file_path不为None，则读入文件后生成为二分图
            
        file_path (str): 文件路径(默认为None)
        
        return: edge_index, edge_attr, x_s, x_t, norm
        """
        if file_path is not None:
            self.read(file_path)
        A = sp.csr_matrix(self.getA())

        data = A.data
        indices = A.indices
        indptr = A.indptr
        var = self.getVars()
        constr = self.getConstrs()
        m, n = A.shape
        x_s = []
        x_t = []
        edge_attr = []
        edge_index = [[], []]
        for i in range(n):
            varType = 1 if var[i].getType() in [COPT.BINARY, COPT.INTEGER] else 0
            varLB = var[i].LB
            varUB = var[i].UB
            hasLB = 1
            hasUB = 1
            if varLB == -COPT.INFINITY:
                varLB = 0
                hasLB = 0
            if varUB == COPT.INFINITY:
                varUB = 0
                hasUB = 0
            obj = var[i].Obj

            assert -COPT.INFINITY < varLB < COPT.INFINITY
            assert -COPT.INFINITY < varUB < COPT.INFINITY
            assert -COPT.INFINITY < obj < COPT.INFINITY
            norm_var_lis = [varLB, varUB,1]
            maxscale = 2*np.sign(norm_var_lis[np.argmax(np.abs(norm_var_lis))])*np.max(np.abs(norm_var_lis))
            ls=[1, hasLB, hasUB, varLB/maxscale, varUB/maxscale, obj, varType, maxscale]#FIXME 还要放入maxscale吗？
            x_t.append(ls)

        for i in range(m):
            start_idx = indptr[i]
            end_idx = indptr[i + 1]
            # col_indices = indices[start_idx:end_idx]
            LB = constr[i].LB
            UB = constr[i].UB
            
            if LB == -COPT.INFINITY and UB < COPT.INFINITY:
                max_item = abs(UB)
                max_sgn = np.sign(UB)
                tmp=[]
                for j in range(start_idx, end_idx):
                    if data[j] == 0:
                        continue
                    if abs(data[j]) > max_item:
                        max_item = abs(data[j])
                        max_sgn = np.sign(data[j])
                    tmp.append(data[j])
                    edge_index[0].append(i)
                    edge_index[1].append(indices[j])  
                max_item *= 2*max_sgn
                # if max_item == 0:
                #     max_item = 1
                edge_attr+=[i/max_item for i in tmp]
                x_s.append([0, 0, 0, 1, UB/max_item])
                
            elif LB > -COPT.INFINITY and UB == COPT.INFINITY:
                max_item = abs(LB)
                max_sgn = np.sign(LB)
                tmp=[]
                for j in range(start_idx, end_idx):
                    if data[j] == 0:
                        continue
                    if abs(data[j]) > max_item:
                        max_item = abs(data[j])
                        max_sgn = np.sign(data[j])
                    tmp.append(data[j])
                    edge_index[0].append(i)
                    edge_index[1].append(indices[j])  
                max_item *= 2*max_sgn
                # if max_item == 0:
                #     max_item = 1
                edge_attr+=[i/max_item for i in tmp]
                x_s.append([0, 1, LB/max_item, 0, 0])
                
            elif LB > -COPT.INFINITY and UB < COPT.INFINITY:
                max_item = max(abs(UB),abs(LB))
                ls = [UB, LB]
                max_sgn = np.sign(ls[np.argmax([abs(UB),abs(LB)])])
                tmp=[]
                for j in range(start_idx, end_idx):
                    if data[j] == 0:
                        continue
                    if abs(data[j]) > max_item:
                        max_item = abs(data[j])
                        max_sgn = np.sign(data[j])
                    tmp.append(data[j])
                    edge_index[0].append(i)
                    edge_index[1].append(indices[j])  
                max_item *= 2*max_sgn
                if max_item == 0:
                    max_item = 1
                edge_attr+=[i/max_item for i in tmp]
                x_s.append([0, 1, LB/max_item, 1, UB/max_item])
                
            else:
                raise ValueError('Wrong constr')
            
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        x_s = torch.tensor(x_s, dtype=torch.float32)
        x_t = torch.tensor(x_t, dtype=torch.float32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)

        norm={}
        norm['varMaxScale'] = x_t[:, -1].abs().max()*2
        norm['maxObj'] = x_t[:, 5].abs().max()*2
        
        return edge_index, edge_attr, x_s, x_t, norm
    
    def generBiparNoN(self, file_path=None):
        """ No Normalization
            生成二分图的各组件，返回值用于生成二分图
            若file_path不为None，则读入文件后生成为二分图
            
        file_path (str): 文件路径(默认为None)
        
        return: edge_index, edge_attr, x_s, x_t, norm
        """
        if file_path is not None:
            self.read(file_path)
        A = sp.csr_matrix(self.getA())

        data = A.data
        indices = A.indices
        indptr = A.indptr
        var = self.getVars()
        constr = self.getConstrs()
        m, n = A.shape
        x_s = []
        x_t = []
        edge_attr = []
        edge_index = [[], []]
        for i in range(n):
            varType = 1 if var[i].getType() in [COPT.BINARY, COPT.INTEGER] else 0
            varLB = var[i].LB
            varUB = var[i].UB
            hasLB = 1
            hasUB = 1
            if varLB == -COPT.INFINITY:
                varLB = 0
                hasLB = 0
            if varUB == COPT.INFINITY:
                varUB = 0
                hasUB = 0
            obj = var[i].Obj

            assert -COPT.INFINITY < varLB < COPT.INFINITY
            assert -COPT.INFINITY < varUB < COPT.INFINITY
            assert -COPT.INFINITY < obj < COPT.INFINITY
            
            ls=[1, hasLB, hasUB, varLB, varUB, obj, varType]#FIXME 还要放入maxscale吗？
            x_t.append(ls)

        for i in range(m):
            start_idx = indptr[i]
            end_idx = indptr[i + 1]
            # col_indices = indices[start_idx:end_idx]
            LB = constr[i].LB
            UB = constr[i].UB
            
            if LB == -COPT.INFINITY and UB < COPT.INFINITY:
                tmp=[]
                for j in range(start_idx, end_idx):
                    if data[j] == 0:
                        continue
                    tmp.append(data[j])
                    edge_index[0].append(i)
                    edge_index[1].append(indices[j])  
                # if max_item == 0:
                #     max_item = 1
                edge_attr+=[i for i in tmp]
                x_s.append([0, 0, 0, 1, UB])
                
            elif LB > -COPT.INFINITY and UB == COPT.INFINITY:
                tmp=[]
                for j in range(start_idx, end_idx):
                    if data[j] == 0:
                        continue
                    tmp.append(data[j])
                    edge_index[0].append(i)
                    edge_index[1].append(indices[j])  
                # if max_item == 0:
                #     max_item = 1
                edge_attr+=[i for i in tmp]
                x_s.append([0, 1, LB, 0, 0])
                
            elif LB > -COPT.INFINITY and UB < COPT.INFINITY:
                ls = [UB, LB]
                tmp=[]
                for j in range(start_idx, end_idx):
                    if data[j] == 0:
                        continue
                    tmp.append(data[j])
                    edge_index[0].append(i)
                    edge_index[1].append(indices[j])  
                edge_attr+=[i for i in tmp]
                x_s.append([0, 1, LB, 1, UB])
                
            else:
                raise ValueError('Wrong constr')
            
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        x_s = torch.tensor(x_s, dtype=torch.float32)
        x_t = torch.tensor(x_t, dtype=torch.float32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)

        norm={}
        norm['varMaxScale'] = 1
        norm['maxObj'] = 1
        
        return edge_index, edge_attr, x_s, x_t, norm