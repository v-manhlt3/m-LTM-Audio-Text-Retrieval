#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

"""
Evaluation tools adapted from https://github.com/fartashf/vsepp/blob/master/evaluation.py
"""

import numpy as np
import torch
import random
from sentence_transformers import util
from loguru import logger
from tools.file_io import load_pickle_file
from gensim.models.word2vec import Word2Vec
import ot
from tqdm import tqdm


def setup_seed(seed):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def align_word_embedding(words_list_path, model_path, nhid):

    words_list = load_pickle_file(words_list_path)
    w2v_model = Word2Vec.load(model_path)
    ntoken = len(words_list)
    weights = np.zeros((ntoken, nhid))
    for i, word in enumerate(words_list):
        if word in w2v_model.wv.index_to_key:
            embedding = w2v_model.wv[word]
            weights[i] = embedding
    weights = torch.from_numpy(weights).float()
    return weights


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


# evaluation tools
def a2t(audio_embs, cap_embs, return_ranks=False, use_ot=False, use_cosine=True):
    # audio to caption retrieval
    num_audios = int(audio_embs.shape[0] / 5)
    index_list = []

    ranks = np.zeros(num_audios)
    top1 = np.zeros(num_audios)
    mAP10 = np.zeros(num_audios)
    for index in tqdm(range(num_audios)):
        # get query audio
        audio = audio_embs[5 * index].reshape(1, audio_embs.shape[1]) # size of [1, audio_emb]
        # print("len audio: ", len(audio))
        # print("len audio embes: ", len(audio_embs))
        # compute scores
        if use_ot:
            batch_size = len(audio_embs)
            a = torch.ones(1)
            b = torch.ones(batch_size)/batch_size
            a = a.to(torch.device("cuda"))
            b = b.to(torch.device("cuda"))
            # print("using ot")

            if use_cosine:
                M_dist = util.cos_sim(audio, cap_embs)
                M_dist = 1 - M_dist
            else:
                M_dist = ot.dist(torch.tensor(audio).to(torch.device("cuda")), torch.tensor(cap_embs).to(torch.device("cuda")))
            # M_dist = M_dist.to(torch.device("cuda"))
            M_dist = M_dist /M_dist.max()
            M_dist = torch.tensor(M_dist).to(torch.device("cuda"))

            d = ot.partial.entropic_partial_wasserstein(a, b, M_dist, reg=0.05, m=0.001,numItermax=10).squeeze(0).cpu().numpy()
            # print("optimal plan matrix: ", d.shape)
        else:
            d = util.cos_sim(torch.Tensor(audio), torch.Tensor(cap_embs)).squeeze(0).numpy() # size of [1, #captions]
        inds = np.argsort(d)[::-1] # sorting metric scores
        index_list.append(inds[0])

        inds_map = []

        rank = 1e20
        #########################################################################
        # find the best rank among five captions
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
            if tmp < 10:
                inds_map.append(tmp + 1)
        ##########################################################################
        inds_map = np.sort(np.array(inds_map))
        if len(inds_map) != 0:
            mAP10[index] = np.sum((np.arange(1, len(inds_map) + 1) / inds_map)) / 5
        else:
            mAP10[index] = 0.
        ranks[index] = rank
        top1[index] = inds[0]
    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(mAP10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr
    

def t2a(audio_embs, cap_embs, return_ranks=False, use_ot=False, use_cosine=True):
    num_audios = int(audio_embs.shape[0] / 5)
    # print("num audios")
    audios = np.array([audio_embs[i]for i in range(0, audio_embs.shape[0], 5)])

    ranks = np.zeros(5 * num_audios)
    top1 = np.zeros(5 * num_audios)

    for index in tqdm(range(num_audios)):

        # get query captions
        queries = cap_embs[5 * index: 5 * index + 5]

        # compute scores
        if use_ot:
            # print("using ot")
            batch_size = len(audios)
            a = torch.ones(1)
            b = torch.ones(batch_size)/batch_size
            a = a.to(torch.device("cuda"))
            b = b.to(torch.device("cuda"))
            d = []
            for query in queries:
                if use_cosine:
                    M_dist = util.cos_sim(query, audios)
                    M_dist = 1 - M_dist
                else:
                    # print(torch.tensor(query).shape)
                    # print(torch.tensor(audios).shape)
                    # print("Evaluation**********************************************")
                    M_dist = ot.dist(torch.tensor(query).unsqueeze(0).to(torch.device("cuda")), torch.tensor(audios).to(torch.device("cuda")))
                M_dist = M_dist /M_dist.max()
                M_dist = torch.tensor(M_dist).to(torch.device("cuda"))

                pi = ot.partial.entropic_partial_wasserstein(a, b, M_dist, reg=0.05, m=0.001,numItermax=10).cpu().numpy()
                d.append(pi)
            d = np.vstack(d)
            # print("optimal plan matrix: ", d.shape)
        else:
            # print("using cosine")
            d = util.cos_sim(torch.Tensor(queries), torch.Tensor(audios)).numpy() # size of [5 queries, #audios]
            # print("cosine matrix shape: ", d.shape)
        # print("d matrix shape: ", d.shape)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr


def a2t_ot(audio_embs, cap_embs, use_ot=False):
    # num_audios = int(audio_embs.shape[0]/5)
    # index_list = []
    
    # audio = [audio_embs[i] for i in range(0, len(audio_embs), 5)]
    # audio = torch.stack(audio).to(torch.device("cuda"))
    # cap_embs = torch.Tensor(cap_embs).to(torch.device("cuda"))
    # audio_embs = torch.Tensor(audio_embs).to(torch.device("cuda"))

    a = torch.ones(audio_embs.size(0))/audio_embs.size(0)
    b = torch.ones(cap_embs.size(0))/cap_embs.size(0)
    a = a.to(torch.device("cuda"))
    b = b.to(torch.device("cuda"))

    M_dist = ot.dist(audio_embs, cap_embs)
    M_dist = M_dist/M_dist.max()
    ground_truth = torch.arange(start=0, end=cap_embs.size(0)).view(-1,1).to(audio_embs.device)

    if use_ot:
        d = ot.partial.entropic_partial_wasserstein(a,b,M_dist, reg=0.04, m=0.94, numItermax=20)
    else:
        d = util.cos_sim(audio_embs, cap_embs)
    sorting = torch.argsort(d, descending=True)
    preds = torch.where(sorting==ground_truth)[1]
    preds = preds.detach().cpu().numpy()
    # print("sorting: ", sorting)
    # print("preds: ", preds)
    # print("ranking: ", ranking)

    medr = np.floor(np.median(preds)) + 1
    meanr = preds.mean() + 1
    r1 = np.mean(preds < 1)
    r5 = np.mean(preds < 5)
    r10 = np.mean(preds < 10)
    r50 = np.mean(preds < 50)
    return r1, r5, r10, r50, medr, meanr

def t2a_ot(audio_embs, cap_embs, use_ot=False, use_cosine=True):
    # cap_embs = torch.Tensor(cap_embs).to(torch.device("cuda"))
    # audio_embs = torch.Tensor(audio_embs).to(torch.device("cuda"))

    a = torch.ones(cap_embs.size(0))/audio_embs.size(0)
    b = torch.ones(audio_embs.size(0))/cap_embs.size(0)
    a = a.to(torch.device("cuda"))
    b = b.to(torch.device("cuda"))

    if use_cosine:
        M_dist = util.cos_sim(audio_embs, cap_embs)
        M_dist = 1 - M_dist
    else:
        M_dist = ot.dist(torch.Tensor(cap_embs).to(torch.device("cuda")), torch.Tensor(audio_embs).to(torch.device("cuda")))
    M_dist = M_dist/M_dist.max()
    ground_truth = torch.arange(start=0, end=cap_embs.size(0)).view(-1,1).to(audio_embs.device)

    if use_ot:  
        d = ot.partial.entropic_partial_wasserstein(a,b,M_dist, reg=0.04, m=0.94, numItermax=20)
        # d = ot.sinkhorn(a,b,M_dist, reg=0.04, numItermax=100)
        print("using ot")
    else:
        d = util.cos_sim(cap_embs, audio_embs)

    sorting = torch.argsort(d, descending=True)
    preds = torch.where(sorting==ground_truth)[1]
    preds = preds.detach().cpu().numpy()

    medr = np.floor(np.median(preds)) + 1
    meanr = preds.mean() + 1
    r1 = np.mean(preds < 1)
    r5 = np.mean(preds < 5)
    r10 = np.mean(preds < 10)
    r50 = np.mean(preds < 50)
    return r1, r5, r10, r50, medr, meanr

def a2t_ot_full2(audio_embs, cap_embs, use_ot=False, use_cosine=True):

    num_audios = int(audio_embs.shape[0]/5)

    audios = np.array([audio_embs[i] for i in range(0, audio_embs.shape[0], 5)]) # size of [#audio]
    metric ={"r1":0, "r5":0, "r10":0, "mean":0, "median":0}
    final_preds = []
# for ind in range(5):
    # captions = np.array([cap_embs[i+ind] for i in range(0, audio_embs.shape[0], 5)]) # size of [#caps]
    # audios = audios.reshape(num_audios, -1)
    if use_ot:
        a = torch.ones(num_audios)/num_audios
        b = torch.ones(cap_embs.shape[0])/cap_embs.shape[0]
        a = a.to(torch.device("cuda"))
        b = b.to(torch.device("cuda"))

        if use_cosine:
            M_dist = util.cos_sim(audios, cap_embs)
            M_dist = 1 - M_dist
            M_dist = M_dist.to(torch.device("cuda"))
        else:
            M_dist = ot.dist(torch.Tensor(audios).to(torch.device("cuda")), torch.Tensor(cap_embs).to(torch.device("cuda")))
        M_dist = M_dist/M_dist.max()

        # d = ot.partial.entropic_partial_wasserstein(a, b, M_dist, reg=0.04, m=0.94,numItermax=100)
        d = ot.sinkhorn(a,b,M_dist, reg=0.02, numItermax=200)
    else:
        d = util.cos_sim(torch.Tensor(audios), torch.Tensor(cap_embs)).to(torch.device("cuda"))

    # d = filter(d)
    sorting_d = torch.argsort(d, descending=True)
    five_d = filter(sorting_d.cpu().numpy())
    print("five_d shape: ", five_d.shape)
    ranks = np.min(five_d, axis=-1)
    print(ranks)
    # ground_truth = torch.arange(start=0, end=num_audios).view(-1, 1).to(torch.device("cuda"))
    # preds = torch.where(sorting_d==ground_truth)[1]
    # preds = preds.detach().cpu().numpy()
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    # mAP10 = 100.0 * np.sum(mAP10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return r1, r5, r10, r50, meanr, medr

def a2t_ot_full(audio_embs, cap_embs, use_ot=False, use_cosine=False):

    num_audios = int(audio_embs.shape[0]/5)

    audios = np.array([audio_embs[i] for i in range(0, audio_embs.shape[0], 5)]) # size of [#audio]
    metric ={"r1":0, "r5":0, "r10":0, "mean":0, "median":0}
    final_preds = []
    for ind in range(5):
        captions = np.array([cap_embs[i+ind] for i in range(0, audio_embs.shape[0], 5)]) # size of [#caps]
        # audios = audios.reshape(num_audios, -1)
        if use_ot:
            a = torch.ones(num_audios)/num_audios
            b = torch.ones(num_audios)/num_audios
            a = a.to(torch.device("cuda"))
            b = b.to(torch.device("cuda"))

            if use_cosine:
                M_dist = util.cos_sim(audios, captions)
                M_dist = 1 - M_dist
                M_dist = M_dist.to(torch.device("cuda"))
            else:
                M_dist = ot.dist(torch.Tensor(audios).to(torch.device("cuda")), torch.Tensor(captions).to(torch.device("cuda")))
            M_dist = M_dist/M_dist.max()

            # d = ot.partial.entropic_partial_wasserstein(a, b, M_dist, reg=0.04, m=0.94,numItermax=100)
            d = ot.sinkhorn(a,b,M_dist, reg=0.02, numItermax=200)
        else:
            d = util.cos_sim(torch.Tensor(audios), torch.Tensor(captions)).to(torch.device("cuda"))

        sorting_d = torch.argsort(d, descending=True)
        ground_truth = torch.arange(start=0, end=num_audios).view(-1, 1).to(torch.device("cuda"))
        preds = torch.where(sorting_d==ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        medr = np.floor(np.median(preds)) + 1
        meanr = preds.mean() + 1
        r1 = np.mean(preds < 1)
        r5 = np.mean(preds < 5)
        r10 = np.mean(preds < 10)
        r50 = np.mean(preds < 50)
        print("R@1: {}|-- R@5: {}|-- R@10: {}".format(r1, r5,r10))
        metric['r1'] += r1
        metric['r5'] += r5
        metric['r10'] += r10
        metric['mean'] += meanr
        metric['median'] += medr
    return metric['r1']/5, metric['r5']/5, metric['r10']/5, r50, metric['mean']/5, metric['median']/5

def t2a_ot_full(audio_embs, cap_embs, use_ot=False, use_cosine=False):

    num_audios = int(audio_embs.shape[0]/5)

    audios = np.array([audio_embs[i] for i in range(0, audio_embs.shape[0], 5)]) # size of [#audio]
    metric ={"r1":0, "r5":0, "r10":0, "mean":0, "median":0}
    for ind in range(5):
        captions = np.array([cap_embs[i+ind] for i in range(0, audio_embs.shape[0], 5)]) # size of [#caps]
        # audios = audios.reshape(num_audios, -1)
        if use_ot:
            a = torch.ones(num_audios)/num_audios
            b = torch.ones(num_audios)/num_audios
            a = a.to(torch.device("cuda"))
            b = b.to(torch.device("cuda"))

            if use_cosine:
                M_dist = util.cos_sim(audios, captions)
                M_dist = 1 - M_dist
                M_dist = M_dist.to(torch.device("cuda"))
            else:
                M_dist = ot.dist(torch.Tensor(audios).to(torch.device("cuda")), torch.Tensor(captions).to(torch.device("cuda")))
            M_dist = M_dist/M_dist.max()
            
            # d = ot.partial.entropic_partial_wasserstein(a, b, M_dist, reg=0.04, m=0.94,numItermax=100)
            d = ot.sinkhorn(a,b,M_dist, reg=0.04, numItermax=100)
        else:
            d = util.cos_sim(torch.Tensor(captions), torch.Tensor(audios)).to(torch.device("cuda"))
        sorting_d = torch.argsort(d, descending=True)
        ground_truth = torch.arange(start=0, end=num_audios).view(-1, 1).to(torch.device("cuda"))
        preds = torch.where(sorting_d==ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        medr = np.floor(np.median(preds)) + 1
        meanr = preds.mean() + 1
        r1 = np.mean(preds < 1)
        r5 = np.mean(preds < 5)
        r10 = np.mean(preds < 10)
        r50 = np.mean(preds < 50)
        metric['r1'] += r1
        metric['r5'] += r5
        metric['r10'] += r10
        metric['mean'] += meanr
        metric['median'] += medr
    return metric['r1']/5, metric['r5']/5, metric['r10']/5, r50, metric['mean']/5, metric['median']/5


def t2a_ot_sampling(audio_embs, cap_embs,use_ot=True):
    num_audios = int(audio_embs.shape[0]/5)
    # audios = np.array()
    audios = np.array([audio_embs[i] for i in range(0, audio_embs.shape[0], 5)]) # size of [#audio]
    metric ={"r1":0, "r5":0, "r10":0, "mean":0, "median":0}
    mini_batch = 10
    # audios = audios[:(audios.shape(0)//mini_batch)*mini_batch]
    k = 10000
    ind = 0
    for ind in range(5):
        print("loop number: ", ind)
        captions = np.array([cap_embs[i+ind] for i in range(0, audio_embs.shape[0], 5)]) # size of [#caps]
        d_k = []
        # audios_sampling = audios.clone()
        # captions_sampling = captions.clone()
        d = torch.zeros(num_audios, num_audios).to(torch.device("cuda"))
        for i in tqdm(range(k)):
            
            # audios_ind = torch.randperm(num_audios)[:mini_batch]
            caps_ind = torch.randperm(num_audios)[:mini_batch]
            # mini_audios = audios[audios_ind]
            mini_caps = captions[caps_ind]
            a = torch.ones(mini_batch)/mini_batch
            b = torch.ones(num_audios)/num_audios
            a = a.to(torch.device("cuda"))
            b = b.to(torch.device("cuda"))

            M_dist = ot.dist(torch.Tensor(mini_caps).to(torch.device("cuda")), torch.Tensor(audios).to(torch.device("cuda")))
            M_dist = M_dist/M_dist.max()
            
            # mini_d = ot.sinkhorn(a,b,M_dist, reg=0.05, numItermax=10)
            mini_d = ot.partial.entropic_partial_wasserstein(a, b, M_dist, reg=0.04, m=0.01, numItermax=100)
            # print(a.device)
            # print(M_dist.device)
            # M_dist = M_dist.detach().cpu()
            # mini_d = ot.partial.partial_wasserstein(a,b,M_dist, m=.95)
            # print(mini_d.shape)
            # mini_d = mini_d[:-1, :-1]

            for m in range(mini_batch):
                for n in  range(num_audios):
                    d[caps_ind[m], n] += mini_d[m,n]
            # d_k.append(d)
        # app_d = sum(d_k)/k
        d = d/k
        # print("size of app_d: ", app_d.shape)
        sorting_d = torch.argsort(d, descending=True)
        ground_truth = torch.arange(start=0, end=num_audios).view(-1, 1).to(torch.device("cuda"))
        preds = torch.where(sorting_d==ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        medr = np.floor(np.median(preds)) + 1
        meanr = preds.mean() + 1
        r1 = np.mean(preds < 1)
        r5 = np.mean(preds < 5)
        r10 = np.mean(preds < 10)
        r50 = np.mean(preds < 50)
        print("R@1: {}--|R@5: {}--|R@10: {}".format(r1, r5, r10))
        print("*"*70)
        metric['r1'] += r1
        metric['r5'] += r5
        metric['r10'] += r10
        metric['mean'] += meanr
        metric['median'] += medr
    return metric['r1']/5, metric['r5']/5, metric['r10']/5, r50, metric['mean']/5, metric['median']/5
    # return metric['r1'], metric['r5'], metric['r10'], r50, metric['mean'], metric['median']

def t2a_ot_sampling2(audio_embs, cap_embs,use_ot=True):
    num_audios = int(audio_embs.shape[0]/5)
    num_audios = 128*7
    audios = np.array([audio_embs[i] for i in range(0, audio_embs.shape[0], 5)]) # size of [#audio]
    metric ={"r1":0, "r5":0, "r10":0, "mean":0, "median":0}
    mini_batch = 512
    mini_batch_t= 128
    # audios = audios[:(audios.shape[0]//mini_batch)*mini_batch]
    k = 10
    ind = 0
    for ind in range(5):
        print("loop number: ", ind)
        captions = np.array([cap_embs[i+ind] for i in range(0, audio_embs.shape[0], 5)]) # size of [#caps]
        d = torch.zeros(num_audios, num_audios).to(torch.device("cuda"))
        for mini_t in range(7):
            for i in tqdm(range(k)):
                audios_ind = torch.randperm(num_audios)[:mini_batch]
                caps_ind = torch.randperm(num_audios)[:mini_batch]
                mini_audios = audios[mini_t*mini_batch_t:mini_t*mini_batch_t+ mini_batch_t]
                mini_caps = captions[caps_ind]
                a = torch.ones(mini_batch)/mini_batch
                b = torch.ones(mini_batch_t)/mini_batch_t
                # a = a.to(torch.device("cuda"))
                # b = b.to(torch.device("cuda"))

                M_dist = ot.dist(torch.Tensor(mini_caps).to(torch.device("cuda")), torch.Tensor(mini_audios).to(torch.device("cuda")))
                M_dist = M_dist/M_dist.max()
                
                M_dist = M_dist.detach().cpu()
                mini_d = ot.partial.partial_wasserstein(a,b,M_dist, m=.95)

                for m in range(mini_batch_t):
                    for n in  range(mini_batch):
                        print(m)
                        print(audios_ind[n])
                        print(audios_ind.size())
                        print(mini_d.size())
                        d[m+mini_t*mini_batch_t, audios_ind[n]] += mini_d[m,n]

        d = d/k
        # print("size of app_d: ", app_d.shape)
        sorting_d = torch.argsort(d, descending=True)
        ground_truth = torch.arange(start=0, end=num_audios).view(-1, 1).to(torch.device("cuda"))
        preds = torch.where(sorting_d==ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        medr = np.floor(np.median(preds)) + 1
        meanr = preds.mean() + 1
        r1 = np.mean(preds < 1)
        r5 = np.mean(preds < 5)
        r10 = np.mean(preds < 10)
        r50 = np.mean(preds < 50)
        print("R@1: {}--|R@5: {}--|R@10: {}".format(r1, r5, r10))
        print("*"*70)
        metric['r1'] += r1
        metric['r5'] += r5
        metric['r10'] += r10
        metric['mean'] += meanr
        metric['median'] += medr
    return metric['r1']/5, metric['r5']/5, metric['r10']/5, r50, metric['mean']/5, metric['median']/5