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


class MahalalobisLoss(nn.Module):

    def __init__(self, epsilon=0.05, reg=0.1, m=0.95, L=None, pot=False):
        super(MahalalobisLoss, self).__init__()
        self.epsilon = epsilon
        self.reg = reg
        self.mmd_reg = MMDLoss()
        self.mmd_reg.cuda()
        self.m =m
        self.POT = pot

    def forward(self, audio_emb, text_emb, M):
        batch_size = audio_emb.size(0)
        a = torch.ones(batch_size)/batch_size
        b = torch.ones(batch_size)/batch_size
        a = a.to(audio_emb.device)
        b = b.to(audio_emb.device)

        pi_hat = torch.eye(batch_size).to(audio_emb.device)/(batch_size)

        M = torch.nan_to_num(M)
        u, s, v =torch.svd(M)
        reg = torch.sum(s)

        audio_matrix = audio_emb.unsqueeze(0).repeat(audio_emb.size(0),1,1)
        text_matrix = text_emb.unsqueeze(1).repeat(1, text_emb.size(0), 1)

        pairwise_dist = audio_matrix - text_matrix
        t_pairwise_dist = pairwise_dist.transpose(1,2)
        M_dist = torch.einsum("ijk,ikj,kk->ij", pairwise_dist, t_pairwise_dist, M)
        M_dist = torch.sqrt(M_dist)
        M_dist = M_dist/M_dist.max()

        if self.POT:
            pi = ot.partial.entropic_partial_wasserstein(a,b,M_dist, reg=self.epsilon, m=self.m)
        else:
            pi = ot.sinkhorn(a,b,M_dist, reg=self.epsilon)
        
        ot_loss = -pi_hat*torch.log(pi)
        ot_loss = torch.sum(ot_loss)

        loss = ot_loss + self.reg*reg
        loss = ot_loss

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
    
