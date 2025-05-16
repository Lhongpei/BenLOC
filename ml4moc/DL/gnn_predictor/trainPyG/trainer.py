import os

import torch
from torch.nn import MSELoss
from tqdm import tqdm

import wandb
from ml4moc.DL.gnn_predictor.utils.loss import *

mseloss = MSELoss()
#--------------------------------------
# def forward loop
#--------------------------------------
def Predictor(model,dataset,device):
    best_config = {}
    for data in dataset:
        for j,i in enumerate(data.name):
            best_config[str(i)] = torch.argmin(model(data.to(device)).squeeze()[j]).detach().cpu().item()
    return best_config
        
        
def RankListTrainer(Loss, model, dataloader, optimizer, scheduler, device, use_wandb, ep, fold, train_folder = None, weighted: bool = False, mode: str = 'train', reparam = False, pretrain = False, cal_improve = False):
    assert mode in ['train', 'valid', 'test']
    # if pretrain == False:
    #     assert encoder == None
    #     assert reparam == False
    if mode == 'train':
        model.train()
    else:
        model.eval()

    if cal_improve:
        tot_pres_time = 0
        tot_default_time = 0
        tot_predict_time = 0
    
    loss_counts = 0
    min_loss = 100000000
    
    for batch_data in dataloader:
        hasnan = False
        batch_data = batch_data.to(device)
        #print(batch_data.shape,batch_sort.shape)
        pred = model(batch_data)

        label = batch_data.y
        loss = Loss(label, pred)
        #print(pred.argmax(dim=1).squeeze().detach().cpu())
        if cal_improve:
            rows = torch.arange(label.size(0)).long().detach().cpu()
            predict = label[rows,pred.argmin(dim=1).squeeze()].detach().cpu()
            default = label[:, 0].detach().cpu()
            
            tot_pres_time += torch.sum(predict < default)
            tot_default_time += torch.sum(default) 
            tot_predict_time += torch.sum(predict)
            
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        for name, par in list(model.named_parameters()):
            if par.grad is None:
                continue
            if torch.isnan(par.grad).any():
                print(name)
                hasnan = True
        if hasnan:
            raise Exception("NaN in gradient") 
        
        loss_counts += loss.detach().cpu().item()
    n_data = len(dataloader.dataset)

    if cal_improve:
        avg_default_time = tot_default_time / n_data
        avg_pres_time = tot_pres_time / n_data
        avg_predict_time = tot_predict_time / n_data
        improve = (avg_default_time - avg_predict_time) / avg_default_time
        print(f'{mode}-Pres time: {avg_pres_time}')
        print(f'{mode}-Default time: {avg_default_time}')
        print(f'{mode}-Predict time: {avg_predict_time}')
        print(f'{mode}-Improve: {improve}')
    print(f'{mode}-lr: {optimizer.param_groups[0]["lr"]}')

    loss_tot_mean = loss_counts / n_data 
    
    if loss_tot_mean < min_loss and mode == 'valid':
        min_loss = loss_tot_mean
        if train_folder is not None:
            torch.save(model.state_dict(), os.path.join(train_folder,f'fold_{fold}_Model_RankList.pth'))
        
    if mode == 'train':
        scheduler.step()
    if cal_improve:
        result_dict = {f"fold_{fold}_{mode}_loss": loss_tot_mean, f"fold_{fold}_{mode}_pres": avg_pres_time.detach().cpu(), 
                    f"fold_{fold}_{mode}_default": avg_default_time, f"fold_{fold}_{mode}_predict": avg_predict_time.detach().cpu(), 
                    f"fold_{fold}_{mode}_improve": improve.detach().cpu()}
    else:
        result_dict = {f"fold_{fold}_{mode}_loss": loss_tot_mean}
    if use_wandb:
        wandb.log(result_dict) 
    
    
    return result_dict

def RegrTrainer(model, dataloader, report_Dict, optimizer, scheduler, device, use_wandb, ep, fold, train_folder = None, weighted: bool = False, mode: str = 'train', cal_improve = False):
    assert mode in ['train', 'valid', 'test']
    # if pretrain == False:
    #     assert encoder == None
    #     assert reparam == False
    if mode == 'train':
        model.train()
    else:
        model.eval()

    loss_counts = 0
    min_loss = 100000000
    
    time_Dict = report_Dict['time_Dict']
    config_Dict = report_Dict['config_Dict']
    
    # assert all on cpu
    assert all([x.device == torch.device('cpu') for x in time_Dict.values() if isinstance(x, torch.Tensor)])
    assert all([x.device == torch.device('cpu') for x in config_Dict.values() if isinstance(x, torch.Tensor)])
    
    minconfig_Dict, mintime_Dict = {}, {}
    for batch_data in tqdm(dataloader, desc=f'DataLoader of {mode} in Epoch {ep}: '):
        hasnan = False
        batch_data = batch_data.to(device)
        #print(batch_data.shape,batch_sort.shape)
        pred = model(batch_data)
        #print(pred.shape)
        #print(torch.max(pred),torch.min(pred))
        label = batch_data.y
        print(f'pred:{pred.squeeze()}')
        print(f'label:{label}')
        loss = mseloss(label, pred.squeeze())
        # print(loss)
        # print(pred.shape, label.shape)
        # print(pred_norm.max(), pred_norm.min())
        
        for i, name in enumerate(batch_data.name):
            # if name in bigpro_list:
            #     bigpro_Dict[name][config_Dict[tuple(batch_data.config.squeeze()[i].detach().cpu().tolist())]] = pred[i].detach().cpu()
            if mintime_Dict.get(name) is None or mintime_Dict[name] > pred[i].detach().cpu():
                mintime_Dict[name] = pred[i].detach().cpu()
                minconfig_Dict[name] = tuple(batch_data.config.squeeze()[i].detach().cpu().tolist())
            
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        for name, par in list(model.named_parameters()):
            if par.grad is None:
                continue
            if torch.isnan(par.grad).any():
                print(name)
                hasnan = True
        if hasnan:
            raise Exception("NaN in gradient") 
        
        loss_counts += loss.detach().cpu().item()
        
    n_data = len(mintime_Dict.keys())
    if cal_improve:
        default_list = torch.tensor([time_Dict[name][0] for name in mintime_Dict.keys()]).to('cpu')
        predict_list = torch.tensor([time_Dict[name][config_Dict[minconfig_Dict[name]]] for name in mintime_Dict.keys()]).to('cpu')

        # 计算平均值和改进
        avg_default_time = torch.sum(default_list) / n_data
        avg_pres_time = torch.sum(predict_list < default_list).float() / n_data
        avg_predict_time = torch.sum(predict_list)/n_data
        improve = (avg_default_time - avg_predict_time) / avg_default_time
        
        print(f'{mode}-Pres time: {avg_pres_time}')
        print(f'{mode}-Default time: {avg_default_time}')
        print(f'{mode}-Predict time: {avg_predict_time}')
        print(f'{mode}-Improve: {improve}')
    print(f'{mode}-lr: {optimizer.param_groups[0]["lr"]}')

    loss_tot_mean = loss_counts / n_data 
    
    if ep//50 == 0 and mode == 'valid':
        min_loss = loss_tot_mean
        if train_folder is not None:
            torch.save(model.state_dict(), os.path.join(train_folder,f'fold_{fold}_Model_' + 'Regr' + '.pth'))
        
    if mode == 'train':
        scheduler.step()
    if cal_improve:
        result_dict = {f"fold_{fold}_{mode}_loss": loss_tot_mean, f"fold_{fold}_{mode}_pres": avg_pres_time.detach().item(), 
                    f"fold_{fold}_{mode}_default": avg_default_time.detach().item(), f"fold_{fold}_{mode}_predict": avg_predict_time.detach().item(), 
                    f"fold_{fold}_{mode}_improve": improve.detach().item()}
    else:
        result_dict = {f"fold_{fold}_{mode}_loss": loss_tot_mean}
    # assert all on cpu
    assert all([x.device == torch.device('cpu') for x in result_dict.values() if isinstance(x, torch.Tensor)])
    
    if use_wandb:
        wandb.log(result_dict) 
    
    # print(f'BigPro Dict: {bigpro_Dict}')
    return result_dict