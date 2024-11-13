import torch
import torch.nn.functional as F
l1_loss = torch.nn.SmoothL1Loss()

#--------------------------------------
# def loss of listnet
#--------------------------------------
def listnet_loss(y_i, z_i, weighted = False):
    """
    ListNet loss function.
    y_i: Tensor of true scores of shape (batch_size, ranking_size)
    z_i: Tensor of predicted scores of shape (batch_size, ranking_size)
    """

    # Compute softmax probabilities
    P_y_i = F.softmax(- torch.log(y_i + 1) + 10, dim=1)
    log_P_z_i = F.log_softmax(z_i, dim=1)
    
    # Compute the ListNet loss
    if weighted:
        loss = -torch.sum(P_y_i * log_P_z_i, dim=1) * torch.var(y_i, dim=1)
    else:
        loss = -torch.sum(P_y_i * log_P_z_i, dim=1)
    return torch.mean(loss)

def regr_loss(y_i, z_i):
    return torch.mean((torch.log(1 + y_i) - z_i) ** 2)
    
def soft_l1_regr_loss(y_i, z_i, weighted = False):
    #return torch.mean(F.smooth_l1_loss(z_i, torch.log(1 + y_i)-torch.log(1 + y_i[:, 7:8])))
    return torch.mean(F.smooth_l1_loss(z_i, (y_i - y_i[:, 7:8])/ y_i[:, 7:8]))


def rate_log_regr_loss(y_i, z_i, weighted = False):
    """
    RateLogRegr loss function.
    y_i: Tensor of true scores of shape (batch_size, ranking_size)
    z_i: Tensor of predicted scores of shape (batch_size, ranking_size)
    """
    assert y_i.size(-1) == 2
    # Compute the RateLogRegr loss
    rate_log = torch.log(1 + y_i[:,0])-torch.log(1 + y_i[:,1])
    l1_loss = torch.nn.SmoothL1Loss()
    return l1_loss(rate_log.unsqueeze(-1), z_i)

def ridge_loss(y, y_hat, alpha=0, weighted=False):  # 默认alpha为0，表示不应用正则化
    if weighted:
        mse = torch.mean((y - y_hat) ** 2, dim=0) * torch.var(y, dim=1)
    else:
        mse = torch.mean((y - y_hat) ** 2, dim=0)
    regularization = alpha * torch.mean(y_hat ** 2)
    return torch.mean(mse) + regularization

