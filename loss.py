import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

         
class DCLoss(nn.Module):
    def __init__(self, num=2):
        super(DCLoss, self).__init__()
        self.num = num
        self.fc1 = nn.Sequential(nn.Linear(2048, 256, bias=False)) 
        self.fc2 = nn.Sequential(nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 256), nn.BatchNorm1d(256))
        
    def forward(self, x):
        x = F.normalize(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.normalize(x)
        loss = 0
        num = int(x.size(0) / self.num)
        for i in range(self.num):
            for j in range(self.num):
                if i<j:
                    loss += ((x[i*num:(i+1)*num,:] - x[j*num:(j+1)*num,:]).norm(dim=1,keepdim=True)).mean()
        return loss

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

class TriLoss(nn.Module):
    def __init__(self, batch_size, margin=0.3):
        super(TriLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        
        vis, vtm, nir, ntm = torch.chunk(inputs, 4, 0)
        
        input1 = torch.cat((vis, ntm), 0)
        n = input1.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist_vm = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_vm = dist_vm + dist_vm.t()
        dist_vm.addmm_(1, -2, input1, input1.t())
        dist_vm = dist_vm.clamp(min=1e-12).sqrt()  # for numerical stability
        
        
        input2 = torch.cat((vtm, nir), 0)
        
        # Compute pairwise distance, replace by the official when merged
        dist_nm = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_nm = dist_nm + dist_nm.t()
        dist_nm.addmm_(1, -2, input2, input2.t())
        dist_nm = dist_vm.clamp(min=1e-12).sqrt()  # for numerical stability
        

        input3 = torch.cat((vis, nir), 0)
        
        # Compute pairwise distance, replace by the official when merged
        dist_vn = torch.pow(input3, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_vn = dist_vn + dist_vn.t()
        dist_vn.addmm_(1, -2, input3, input3.t())
        dist_vn = dist_vn.clamp(min=1e-12).sqrt()  # for numerical stability
        
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        dist_ap1, dist_an1 = [], []
        for i in range(n):
            dist_ap1.append(dist_vn[i][mask[i]].mean().unsqueeze(0))
            dist_an1.append(dist_vn[i][mask[i]==0].mean().unsqueeze(0))
        dist_ap1 = torch.cat(dist_ap1)
        dist_an1 = torch.cat(dist_an1)
        dist_ap1 = dist_ap1.mean()
        dist_an1 = dist_an1.mean()
        

        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist_vm[i][mask[i]].mean().unsqueeze(0))
            dist_an2.append(dist_vm[i][mask[i]==0].mean().unsqueeze(0))
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)
        dist_ap2 = dist_ap2.mean()
        dist_an2 = dist_an2.mean()
        

        dist_ap3, dist_an3 = [], []
        for i in range(n):
            dist_ap3.append(dist_nm[i][mask[i]].mean().unsqueeze(0))
            dist_an3.append(dist_nm[i][mask[i]==0].mean().unsqueeze(0))
        dist_ap3 = torch.cat(dist_ap3)
        dist_an3 = torch.cat(dist_an3)
        dist_ap3 = dist_ap3.mean()
        dist_an3 = dist_an3.mean()
        
        #print(dist_ap1.mean(), dist_ap2.mean(), dist_ap3.mean())
        #print(dist_an1.mean(), dist_an2.mean(), dist_an3.mean())

        if  dist_ap2 > dist_ap3:
        
            loss1 = torch.abs(dist_ap2 - dist_ap3.detach())# + dist_an2.detach() - dist_an3
        else:
            loss1 = torch.abs(dist_ap2.detach() - dist_ap3)# + dist_an2.detach() - dist_an3

        return loss1# + loss2



        
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx