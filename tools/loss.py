#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import torch
import torch.nn as nn
from sentence_transformers import util
import torch.nn.functional as F
import ot

import torch
from torch import nn
# from tools.random_prj import sliced_Wasserstein
from tools.mmd import mix_rbf_mmd2

sigma_list = [1, 2, 4, 8, 16]
eps = 1e-8
def gaussian_dotprod_kernel(x, y):
    k_xx = torch.pow(x@x.t(), 2)
    k_yy = torch.pow(y@y.t(), 2)
    k_xy = 2*torch.pow(x@y.t(), 2)
    gau_kernel = torch.exp(-0.5*(k_xx + k_yy - k_xy))
    return gau_kernel

class TripletLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, audio_embeds, text_embeds, labels):
        """

        :param audio_embeds:
        :param text_embeds:
        :param labels:
        :return:
        """

        n = audio_embeds.size(0)  # batch size

        # dist = []
        sim_a2t = util.cos_sim(audio_embeds, text_embeds)  # (batch_size, x batch_size)
        sim_ap = torch.diag(sim_a2t).view(n, 1)
        d1 = sim_ap.expand_as(sim_a2t)
        d2 = sim_ap.t().expand_as(sim_a2t)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = F.relu(self.margin + sim_a2t - d1)
        # compare every diagonal score to scores in its row
        # audio retrieval
        cost_a = F.relu(self.margin + sim_a2t - d2)

        # clear diagonals
        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(cost_a.device)
        cost_s = cost_s.masked_fill(mask, 0)
        cost_a = cost_a.masked_fill(mask, 0)

        cost_s = cost_s.max(1)[0]
        cost_a = cost_a.max(0)[0]

        loss = (cost_s.sum() + cost_a.sum()) / n

        return loss


class BiDirectionalRankingLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(BiDirectionalRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, audio_embeds, text_embeds, labels):
        """

        :param audio_embeds: (batch_size, embed_dim)
        :param text_embeds: (batch_size, embed_dim)
        :param labels: (batch_size, )
        :return:
        """

        n = audio_embeds.size(0)  # batch size

        # dist = []
        sim_a2t = util.cos_sim(audio_embeds, text_embeds)  # (batch_size, x batch_size)
        sim_ap = torch.diag(sim_a2t).view(n, 1)
        d1 = sim_ap.expand_as(sim_a2t)
        d2 = sim_ap.t().expand_as(sim_a2t)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = F.relu(self.margin + sim_a2t - d1)
        # compare every diagonal score to scores in its row
        # audio retrieval
        cost_a = F.relu(self.margin + sim_a2t - d2)

        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(cost_a.device)

        cost_s = cost_s.masked_fill(mask, 0)
        cost_a = cost_a.masked_fill(mask, 0)

        loss = (cost_s.sum() + cost_a.sum()) / n

        return loss


class NTXent(nn.Module):

    def __init__(self, temperature=0.07, epsilon=0.1):
        super(NTXent, self).__init__()
        self.loss = nn.LogSoftmax(dim=1)
        self.tau = temperature
        self.epsilon = epsilon

    def forward(self, audio_embeds, text_embeds, labels):

        n = batch_size = audio_embeds.shape[0]

        a2t = util.cos_sim(audio_embeds, text_embeds) / self.tau
        t2a = util.cos_sim(text_embeds, audio_embeds) / self.tau

        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(a2t.device)
        mask_diag = mask.diag()
        mask_diag = torch.diag_embed(mask_diag)
        mask = mask ^ mask_diag

        a2t_loss = - self.loss(a2t).masked_fill(mask, 0).diag().mean()
        t2a_loss = - self.loss(t2a).masked_fill(mask, 0).diag().mean()
        
        prob_a2t = torch.nn.functional.softmax(a2t, dim=-1)
        ent_a2t = torch.mean(torch.sum(prob_a2t*torch.log(prob_a2t), dim=-1))
        
        prob_t2a = torch.nn.functional.softmax(t2a, dim=-1)
        ent_t2a = torch.mean(torch.sum(prob_t2a*torch.log(prob_t2a), dim=-1))

        ent_reg = self.epsilon*(ent_a2t + ent_t2a)
        # print("Entropy reg: ", ent_reg)
        # loss = 0.5 * a2t_loss + 0.5 * t2a_loss - ent_reg
        loss = 0.5 * a2t_loss + 0.5 * t2a_loss

        return loss


class WeightTriplet(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2):
        super(WeightTriplet, self).__init__()
        self.margin = margin

    def polyloss(self, sim_mat, label):
        epsilon = 1e-5
        size = sim_mat.size(0)
        hh = sim_mat.t()

        loss = list()
        for i in range(size):
            pos_pair_ = sim_mat[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)
            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)

            loss.append(pos_loss + neg_loss)
        for i in range(size):
            pos_pair_ = hh[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = hh[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)

            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / size
        return loss

    def forward(self, audio_embeds, text_embeds, labels):
        # compute image-sentence score matrix
        scores = util.cos_sim(audio_embeds, text_embeds)
        loss = self.polyloss(scores, labels)
        return loss



class POTLoss(nn.Module):

    def __init__(self, epsilon=0.05, m=0.95, use_cosine=False, tau=0.7):
        super(POTLoss, self).__init__()
        self.epsilon = epsilon
        self.m = m
        self.use_cosine = use_cosine
        self.temp = tau
        # self.metric = metric
    
    def forward(self, audio_emb, text_emb, labels):
        batch_size = audio_emb.size(0)
        a = torch.ones(batch_size)/batch_size
        b = torch.ones(batch_size)/batch_size
        a = a.to(audio_emb.device)
        b = b.to(audio_emb.device)
        # print("labels: ", labels)
        true_label = torch.arange(batch_size).to(torch.int64).to(audio_emb.device)

        if self.use_cosine:
            M_dist = util.cos_sim(audio_emb, text_emb) 
            M_dist = 1 - M_dist
        else:
            M_dist = ot.dist(audio_emb, text_emb)
        M_dist = M_dist /M_dist.max()
        # pi = ot.sinkhorn(a,b,M_dist, reg=self.epsilon, numItermax=10)
        pi = ot.partial.entropic_partial_wasserstein(a,b,M_dist, reg=self.epsilon, m=self.m)
        # pi = ot.partial.partial_wasserstein(a=a, b=b, M=M_dist.to(audio_emb.device), m=self.m).to(audio_emb.device)

        loss = F.cross_entropy(pi, true_label)
        return loss

class OTLoss(nn.Module):
    def __init__(self, epsilon=0.05, use_cosine=True):
        super(OTLoss, self).__init__()
        self.epsilon = epsilon
        self.kl_loss = nn.KLDivLoss()
        self.use_cosine = use_cosine

    def forward(self, audio_emb, text_emb, labels):
        batch_size = audio_emb.size(0)
        a = torch.ones(batch_size)/batch_size
        b = torch.ones(batch_size)/batch_size
        a = a.to(audio_emb.device)
        b = b.to(audio_emb.device)
        # print("labels: ", labels)
        true_label = torch.arange(batch_size).to(torch.int64).to(audio_emb.device)
        pi_hat = torch.eye(batch_size).to(audio_emb.device)/(batch_size)
        # uniform_label = torch.ones(batch_size, batch_size) - torch.eye(batch_size)
        # uniform_label = uniform_label.to(audio_emb.device)/(batch_size*batch_size - batch_size)

        if self.use_cosine:
            M_dist = util.cos_sim(audio_emb, text_emb) 
            M_dist = 1 - M_dist
        else:
            M_dist = ot.dist(audio_emb, text_emb)
        
        M_dist = M_dist / M_dist.max()

        pi = ot.sinkhorn(a,b,M_dist, reg=self.epsilon, numItermax=100)
        # pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a,b,M_dist, 0.1, 10, numItermax=10)
        # pi = ot.unbalanced.mm_unbalanced(a,b,M_dist, 1.0)

        loss = F.cross_entropy(pi, true_label)
        # loss = -pi_hat*torch.log(pi)
        loss = torch.sum(loss)

        final_loss = loss 

        return final_loss

class WassersteinLoss(nn.Module):

    def __init__(self, epsilon=0.05, use_cosine=True, reg=0.1):
        super(WassersteinLoss, self).__init__()
        self.epsilon = epsilon
        self.use_cosine = use_cosine
        self.kl_loss = nn.KLDivLoss()
        self.reg = reg
        # self.metric = metric
    
    def forward(self, audio_emb, text_emb, labels):
        batch_size = audio_emb.size(0)
        a = torch.ones(batch_size)/batch_size
        b = torch.ones(batch_size)/batch_size
        # a = a.to(audio_emb.device)
        # b = b.to(audio_emb.device)

        a1 = torch.ones(batch_size//2)/(batch_size//2)
        b1 = torch.ones(batch_size//2)/(batch_size//2)
        a1 = a1.to(audio_emb.device)
        b1 = b1.to(audio_emb.device)
        # print("labels: ", labels)
        true_label = torch.arange(batch_size).to(torch.int64).to(audio_emb.device)

        if self.use_cosine:
            M_dist = util.cos_sim(audio_emb, text_emb) 
            M_dist = 1 - M_dist
        else:
            M_dist = ot.dist(audio_emb, text_emb)
        M_dist = M_dist /M_dist.max()
        # l2_dist = ot.dist(audio_emb, text_emb)
        # l2_dist = l2_dist /l2_dist.max()
        pi = ot.sinkhorn(a,b,M_dist, reg=self.epsilon, numItermax=10)
        

        loss = F.cross_entropy(pi, true_label) + self.reg*(self.kl_loss(a, torch.sum(pi, 0)) + self.kl_loss(b, torch.sum(pi, 1)))
        return loss
    

def compute_distance(audio, text, M):
    dist = torch.zeros(audio.size(0), text.size(0)).to(audio.device)
    for i in range(audio.size(0)):
        for j in range(text.size(0)):
            dist[i,j] = (audio[i]-text[j])@M@(audio[i]- text[j]).t()
    return dist

class MahalalobisL(nn.Module):

    def __init__(self, epsilon=0.05, use_cosine=True, reg=0.1):
        super(MahalalobisL, self).__init__()
        self.epsilon = epsilon
        self.use_cosine = use_cosine
        self.kl_loss = nn.KLDivLoss()
        self.reg = reg
        self.mmd_reg = MMDLoss()
        self.mmd_reg.cuda()
        # self.metric = metric
    
    def forward(self, audio_emb, text_emb, L):
        batch_size = audio_emb.size(0)
        a = torch.ones(batch_size)/batch_size
        b = torch.ones(batch_size)/batch_size
        a = a.to(audio_emb.device)
        b = b.to(audio_emb.device)

        true_label = torch.arange(batch_size).to(torch.int64).to(audio_emb.device)
        pi_hat = torch.eye(batch_size).to(audio_emb.device)/(batch_size)

        # diagonal matrix
        # L = torch.clamp(L, min=0)
        M = torch.diag(L)
        reg = torch.sum(L)
        neg_eigen = L>0
        # print("M shape: ", M.size())

        # Gram matrix
        # M = L
        # u,s,v = torch.svd(M)
        # s = torch.clamp(s, min=0)
        # reg = torch.sum(s)
        # neg_eigen = s>0

        if not self.use_cosine:
            # Mahanalobis distance
            pairwise_dist = audio_emb.unsqueeze(0).repeat(audio_emb.size(0),1,1) - text_emb.unsqueeze(1).repeat(1, text_emb.size(0), 1)
            t_pairwise_dist = pairwise_dist.transpose(1,2)
            M_dist = torch.einsum("ijk,ikj,kk->ij", pairwise_dist, t_pairwise_dist, M)
            M_dist = torch.sqrt(M_dist)
        else:
            # fulrank affine matrix
            M_dist = torch.einsum("ik,jk,kk->ij", audio_emb, text_emb, M)

        M_dist = M_dist/M_dist.max()

        # pi = ot.sinkhorn(a,b,M_dist, reg=self.epsilon, numItermax=100)
        pi = ot.partial.entropic_partial_wasserstein(a,b,M_dist, reg=self.epsilon, m=0.8, numItermax=10)
        # loss = F.cross_entropy(pi, true_label) 
        wloss = -pi_hat*torch.log(pi)
        wloss = torch.sum(wloss)

        # mmd_reg = self.mmd_reg(audio_emb, text_emb)
        # loss = wloss + self.reg*torch.min(reg-30, torch.tensor(0))
        # loss = wloss + self.reg*reg
        loss = wloss
        # loss = wloss + self.reg*mmd_reg

        return loss

class MahalalobisL2(nn.Module):

    def __init__(self, epsilon=0.05, use_cosine=True, reg=0.1, m=0.95, L=None):
        super(MahalalobisL2, self).__init__()
        self.epsilon = epsilon
        self.use_cosine = use_cosine
        self.kl_loss = nn.KLDivLoss()
        self.reg = reg
        self.mmd_reg = MMDLoss()
        self.mmd_reg.cuda()
        self.m =m
        self.L = L
        # self.
        # self.metric = metric
    def set_L(self, L):
        self.L =L
    def forward(self, audio_emb, text_emb):
        batch_size = audio_emb.size(0)
        a = torch.ones(batch_size)/batch_size
        b = torch.ones(batch_size)/batch_size
        a = a.to(audio_emb.device)
        b = b.to(audio_emb.device)

        true_label = torch.arange(batch_size).to(torch.int64).to(audio_emb.device)
        pi_hat = torch.eye(batch_size).to(audio_emb.device)/(batch_size)

        # diagonal matrix
        # L = torch.clamp(L, min=0)
        # M = torch.diag(L)
        M = self.L
        M = torch.nan_to_num(M)
        u, s, v =torch.svd(M)
        reg = torch.sum(s)

        # if not self.use_cosine:
            # Mahanalobis distance
        audio_matrix = audio_emb.unsqueeze(0).repeat(audio_emb.size(0),1,1)
        text_matrix = text_emb.unsqueeze(1).repeat(1, text_emb.size(0), 1)
        # print("Audio matrix shape: ", audio_matrix.shape)
        # print("text matrix shape: ", text_matrix.shape)
        pairwise_dist = audio_matrix - text_matrix
        t_pairwise_dist = pairwise_dist.transpose(1,2)
        # print("Pairwise dist shape: ", pairwise_dist.shape)
        M_dist = torch.einsum("ijk,ikj,kk->ij", pairwise_dist, t_pairwise_dist, M)
        M_dist = torch.sqrt(M_dist)
        M_dist = M_dist/M_dist.max()


        pi = ot.sinkhorn(a,b,M_dist, reg=self.epsilon)
        # print("Pi: ", pi.shape)
        wloss = -pi_hat*torch.log(pi)
        wloss = torch.sum(wloss)
        # print("Wloss: ", wloss)

        loss = wloss + self.reg*reg
        loss = wloss

        return loss

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        # self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels))
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            curr_band_width = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
            print("kernel bandwidth: ", curr_band_width)
            return curr_band_width

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(X.device))[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, bandwidth=None):
        super().__init__()
        self.kernel = RBF()

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    
