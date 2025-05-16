import os

import torch
from tqdm import tqdm

import wandb
from ml4moc.DL.gnn_predictor.utils.loss import *


#--------------------------------------
# def forward loop
#--------------------------------------
def RankListTrainer(model, encoder, dataloader, report_Dict, optimizer, scheduler, device, use_wandb, ep, fold, train_folder = None, weighted: bool = False, mode: str = 'train', reparam = False, pretrain = False):
    assert mode in ['train', 'valid', 'test']
    # if pretrain == False:
    #     assert encoder == None
    #     assert reparam == False
    if mode == 'train':
        model.train()
    else:
        model.eval()

    tot_pres_time = 0
    tot_default_time = 0
    tot_predict_time = 0

    loss_counts = 0
    min_loss = 100000000
    
    time_Dict = report_Dict['time_Dict']
    
    for batch_data in dataloader:
        hasnan = False
        batch_data = batch_data.to(device)
        #print(batch_data.shape,batch_sort.shape)
        if pretrain:
            xs_mu, xs_logsigma, xs_z, xt_mu, xt_logsigma, xt_z = encoder(batch_data.x_s, batch_data.x_t, batch_data.edge_index, batch_data.edge_attr)
            if reparam:
                pred = model(xs_z, xt_z, batch_data.edge_attr, batch_data.edge_index, batch_data.x_s_batch, batch_data.x_t_batch)
            else:
                pred = model(xs_mu, xt_mu, batch_data.edge_attr, batch_data.edge_index, batch_data.x_s_batch, batch_data.x_t_batch)
        else:
            pred = model(batch_data.x_s, batch_data.x_t, batch_data.edge_attr, batch_data.edge_index, batch_data.x_s_batch, batch_data.x_t_batch)

        label = batch_data.y
        #print(label.shape,pred.shape)
        loss = listnet_loss(label, pred, weighted = weighted)
        
        min_index = pred.detach().cpu().argmin(dim=1).squeeze().tolist()
        if type(min_index) == int:
            min_index = [min_index]
        # print(min_index)
        #print(f'min{len(min_index)},name{len(batch_data.name)}')
        predict_list = [time_Dict[name][min_index[i]] for i,name in enumerate(batch_data.name)]
        default_list = [time_Dict[name][0] for name in batch_data.name]
        predict = torch.tensor(predict_list).detach().cpu()
        default = torch.tensor(default_list).detach().cpu()
        # rows = torch.arange(label.size(0)).long()
        # predict = label[rows,pred.argmax(dim=1).squeeze()].cpu()
        # default = label[:, 0].cpu()  
          
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
    
    result_dict = {f"fold_{fold}_{mode}_loss": loss_tot_mean, f"fold_{fold}_{mode}_pres": avg_pres_time.detach().cpu(), 
                   f"fold_{fold}_{mode}_default": avg_default_time, f"fold_{fold}_{mode}_predict": avg_predict_time.detach().cpu(), 
                   f"fold_{fold}_{mode}_improve": improve.detach().cpu()}
    if use_wandb:
        wandb.log(result_dict) 
    
    
    return result_dict

def RegrTrainer(model, encoder, dataloader, report_Dict, optimizer, scheduler, device, use_wandb, ep, fold, train_folder = None, weighted: bool = False, mode: str = 'train', reparam = False, pretrain = False):
    assert mode in ['train', 'valid', 'test']
    # if pretrain == False:
    #     assert encoder == None
    #     assert reparam == False
    if mode == 'train':
        model.train()
    else:
        model.eval()

    tot_pres_time = 0
    tot_default_time = 0
    tot_predict_time = 0

    loss_counts = 0
    min_loss = 100000000
    
    time_Dict = report_Dict['time_Dict']
    config_Dict = report_Dict['config_Dict']
    
    # assert all on cpu
    assert all([x.device == torch.device('cpu') for x in time_Dict.values() if isinstance(x, torch.Tensor)])
    assert all([x.device == torch.device('cpu') for x in config_Dict.values() if isinstance(x, torch.Tensor)])
    
    # bigpro_list = [
    # "train_1600r_1600c_0.01d_instance_1322.mps.gz",
    # "train_1600r_1600c_0.05d_instance_1752.mps.gz",
    # "valid_1600r_800c_0.01d_instance_462.mps.gz",
    # "train_1600r_1600c_0.05d_instance_217.mps.gz",
    # "train_1600r_800c_0.01d_instance_778.mps.gz",
    # "train_1600r_800c_0.05d_instance_931.mps.gz",
    # "test_1600r_1600c_0.01d_instance_128.mps.gz",
    # "train_1600r_800c_0.01d_instance_1066.mps.gz",
    # "train_1600r_800c_0.01d_instance_1280.mps.gz",
    # "test_1600r_1600c_0.01d_instance_349.mps.gz"
    # ]
    # bigpro_Dict = {name: {j:0 for j in range(25)} for name in bigpro_list}
    minconfig_Dict, mintime_Dict = {}, {}
    for batch_data in tqdm(dataloader, desc=f'DataLoader of {mode} in Epoch {ep}: '):
        hasnan = False
        batch_data = batch_data.to(device)
        #print(batch_data.shape,batch_sort.shape)
        if pretrain:
            xs_mu, xs_logsigma, xs_z, xt_mu, xt_logsigma, xt_z = encoder(batch_data.x_s, batch_data.x_t, batch_data.edge_index, batch_data.edge_attr)
            if reparam:
                pred = model(xs_z, xt_z, batch_data.edge_attr, batch_data.edge_index, batch_data.x_s_batch, batch_data.x_t_batch, batch_data.config)
            else:
                pred = model(xs_mu, xt_mu, batch_data.edge_attr, batch_data.edge_index, batch_data.x_s_batch, batch_data.x_t_batch, batch_data.config)
        else:
            pred = model(batch_data.x_s, batch_data.x_t, batch_data.edge_attr, batch_data.edge_index, batch_data.x_s_batch, batch_data.x_t_batch, batch_data.config)
        #print(pred.shape)
        #print(torch.max(pred),torch.min(pred))
        label = batch_data.y
        print(f'pred:{pred.squeeze()}')
        print(f'label:{label}')
        loss = ridge_loss(label, pred.squeeze(), weighted = weighted)
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
    print(f'Number of data: {n_data}')
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
    
    if loss_tot_mean < min_loss and mode == 'valid':
        min_loss = loss_tot_mean
        if train_folder is not None:
            torch.save(model.state_dict(), os.path.join(train_folder,f'fold_{fold}_Model_' + 'Regr' + '.pth'))
        
    if mode == 'train':
        scheduler.step()
    
    result_dict = {f"fold_{fold}_{mode}_loss": loss_tot_mean, f"fold_{fold}_{mode}_pres": avg_pres_time.detach().item(), 
                   f"fold_{fold}_{mode}_default": avg_default_time.detach().item(), f"fold_{fold}_{mode}_predict": avg_predict_time.detach().item(), 
                   f"fold_{fold}_{mode}_improve": improve.detach().item()}
    # assert all on cpu
    assert all([x.device == torch.device('cpu') for x in result_dict.values() if isinstance(x, torch.Tensor)])
    
    if use_wandb:
        wandb.log(result_dict) 
    
    # print(f'BigPro Dict: {bigpro_Dict}')
    return result_dict