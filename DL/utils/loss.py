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

def lambdaLoss(y_i, z_i, delta = 1.0, sigma = 1.0):
    """
    LambdaRank loss function.
    y_i: Tensor of true scores of shape (batch_size, ranking_size)
    z_i: Tensor of predicted scores of shape (batch_size, ranking_size)
    delta: Threshold for relevance
    sigma: Width of the sigmoid function
    """
    # Compute the LambdaRank loss
    S_ij = torch.sign(y_i - y_i.t())
    C_ij = 0.5 * (1.0 - S_ij)
    P_ij = 1.0 / (1.0 + torch.exp(-sigma * (z_i - z_i.t())))
    loss = torch.sum(C_ij * torch.log(1.0 + torch.exp(-delta * (P_ij - 0.5))), dim=1)
    return torch.mean(loss)

def mulClassLoss(y_true, y_pred, weighted = False):
    #print(y_pred)
    """
    使用真实时间最低的项作为真实类别。
    
    y_true_scores: 真实时间，形状为 (batch_size, num_classes)。
    y_pred_scores: 预测概率（logits），形状为 (batch_size, num_classes)。
    """
    # 确定每个样本真实得分最高的类别
    y_true_labels = torch.argmin(y_true, dim=1)
    
    # 使用预测得分（logits）和确定的真实类别计算交叉熵损失
    loss = F.cross_entropy(y_pred, y_true_labels)
    return loss

def weightSortLoss(pred,sort,lossCoef, regression_loss):
    """
    用来计算排序损失
    :param pred: the predicted ranking of methods
    :param data: the ground truth ranking of methods
    :param sort: the sorted ranking of methods
    :param lam: the weight of regularization
    :return: the loss
    """
    loss = 0
    for i in range(len(pred)):
        for j in range(i+1,len(pred)):
            loss += torch.tensor(lossCoef(sort[i],sort[j]))*regression_loss(pred[i],pred[j])
    
    return loss