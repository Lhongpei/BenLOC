import os

import torch
from torch.nn import MSELoss
from tqdm import tqdm

import wandb
from DL.utils.loss import *
import numpy as np
mseloss = MSELoss()
#--------------------------------------
# def forward loop
#--------------------------------------
def predict_loop(model,dataset,device):
    best_config = {}
    config_time = {}
    model.eval()
    for data in dataset:
        for j,i in enumerate(data.name):
            best_config[str(i)] = torch.argmin(model(data.to(device)).squeeze()[j]).detach().cpu().item()
            config_time[str(i)] = data.y[j, best_config[str(i)]].detach().cpu().item()
    return best_config, config_time

def embedding_loop(model, dataset, device):
    graph_embedding = {}
    for data in dataset:
        if isinstance(data.name, str):
            graph_embedding[str(data.name)] = model.graph_embedding(data.to(device)).squeeze().detach().cpu().numpy()
        else:
            for j,i in enumerate(data.name):
                graph_embedding[str(i)] = model.graph_embedding(data.to(device)).squeeze()[j].detach().cpu().numpy()
    return graph_embedding
        
        
def train_loop(Loss, model, dataloader, optimizer, scheduler, device, use_wandb, ep, fold, train_folder = None, weighted: bool = False, mode: str = 'train', cal_improve = False, default_index = 0):
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
        tot_default_time = []
        tot_predict_time = []
    
    loss_counts = 0
    min_loss = 100000000


    
    for batch_data in dataloader:
        hasnan = False
        batch_data = batch_data.to(device)

        #print(batch_data.shape,batch_sort.shape)
        pred = model(batch_data).view(batch_data.num_graphs, -1)
        label = batch_data.y
        label = label.view(batch_data.num_graphs, -1)
        #print(label.shape, pred.shape)
        assert label.shape == pred.shape
        loss = Loss(label, pred)
        #print(batch_data.name)
        #print(pred.argmax(dim=1).squeeze().detach().cpu())
        if cal_improve:
            #print(f'label:{label}')
            best_index = pred.argmin(dim=1).squeeze().detach().cpu()
            #print(f'best_index:{best_index}')
            best_config = label.argmin(dim=1).squeeze().detach().cpu()
            if not torch.numel(best_index) == 1:
                predict = torch.tensor([label[i, best_index[i]] for i in range(len(best_index))])
                #predict = torch.tensor([label[i,0] if pred[i,0] <= 0 else label[i,1] for i in range(len(best_index))])
            else:
                predict = label[0, best_index].detach().cpu()
                #predict = label[0,0].detach().cpu() if pred[0,0] <= 0 else label[0,1].detach().cpu()
            default = label[:, default_index].detach().cpu()
            
            tot_pres_time += torch.sum(best_index == best_config)
            tot_default_time += default.detach().cpu().tolist() if not isinstance(default, float) else [default]
            tot_predict_time += predict.detach().cpu().tolist() if not torch.numel(predict) == 1 else [predict]
            
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # for name, par in list(model.named_parameters()):
            #     if par.grad is None:
            #         continue
            #     print(par.grad.mean())
            
        
        if hasnan:
            raise Exception("NaN in gradient") 
        
        loss_counts += loss.detach().cpu().item()
    n_data = len(dataloader.dataset)

    if cal_improve:
        avg_default_time = np.exp(np.sum(np.log(np.array(tot_default_time)+10))/n_data) - 10
        avg_pres_time = tot_pres_time / n_data
        avg_predict_time = np.exp(np.sum(np.log(np.array(tot_predict_time)+10))/n_data) - 10
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
                    f"fold_{fold}_{mode}_default": avg_default_time, f"fold_{fold}_{mode}_predict": avg_predict_time, 
                    f"fold_{fold}_{mode}_improve": improve, f"fold_{fold}_{mode}_lr": optimizer.param_groups[0]["lr"]}
    else:
        result_dict = {f"fold_{fold}_{mode}_loss": loss_tot_mean}
    if use_wandb:
        wandb.log(result_dict) 
    
    
    return result_dict

