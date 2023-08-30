import numpy as np
import torch
from sklearn import random_projection
from sklearn.random_projection import GaussianRandomProjection
from torch.nn.functional import pad

PRJ_VAL = [-np.sqrt(3), 0.0, np.sqrt(3)]
PROBS = [1/6, 2/3, 1/6]


def init_prj_matrix(d: int, k: int):
    eles = []
    for i in range(d):
        for j in range(k):
            val = np.random.choice(np.array(PRJ_VAL), p=PROBS)
            eles.append(val)
    eles = np.array(eles).reshape(k, d)
    return torch.tensor(eles).float().to(torch.device("cuda"))


def euclidean_prj_dist(list_prj, a, b):
    avg_dist = torch.zeros((a.size(0), a.size(0))).to(a.device)

    for prj in list_prj:
        prj_a = torch.matmul(a, prj.T.to(a.device)) # [B, prj_d]
        prj_b = torch.matmul(b, prj.T.to(a.device)) # [B, prj_d]
        # eucli_dist = torch.norm(prj_a -prj_b, p=2, dim=-1)
        # print(prj_a.size())
        dist = torch.cdist(prj_a, prj_b, p=2)
        # print(dist.size())
        # print("*"*30)
        avg_dist = avg_dist + dist

    return avg_dist/len(list_prj)

def euclidean_prj_dist2(a, b, n_prj=10):
    avg_dist = torch.zeros((a.size(0), a.size(0))).to(torch.device("cuda"))
    dim = a.size(-1)
    gau_prj = GaussianRandomProjection(dim/2)
    a_detach = a.clone()
    b_detach = b.clone()

    for prj in range(n_prj):
        prj_a = gau_prj.fit_transform(a_detach.detach().cpu().numpy()) # [B, prj_d]
        prj_b = gau_prj.fit_transform(b_detach.detach().cpu().numpy()) # [B, prj_d]
        # eucli_dist = torch.norm(prj_a -prj_b, p=2, dim=-1)
        # print(prj_a.size())
        dist = torch.cdist(prj_a, prj_b, p=2)
        # print(dist.size())
        # print("*"*30)
        avg_dist = avg_dist + dist

    return -avg_dist/len(list_prj)

def quantile_function(qs, cws, xs):
    n = xs.shape[0]
    cws = cws.T.contiguous()
    qs = qs.T.contiguous()
    idx = torch.searchsorted(cws, qs, right=False).T
    return torch.gather(xs, 0, torch.clamp(idx, 0, n - 1))

def one_dimensional_Wasserstein(u_values, v_values, u_weights=None, v_weights=None, p=2):
    n = u_values.shape[0]
    m = v_values.shape[0]
    # dist = torch.cdist(u_values, v_values, p=2)
    if u_weights is None:
        u_weights = torch.full(u_values.shape, 1. / n,
                               dtype=u_values.dtype, device=u_values.device)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(
            u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = torch.full(v_values.shape, 1. / m,
                               dtype=v_values.dtype, device=v_values.device)
    elif v_weights.ndim != v_values.ndim:
        v_weights = torch.repeat_interleave(
            v_weights[..., None], v_values.shape[-1], -1)

    u_sorter = torch.sort(u_values, 0)[1]
    u_values = torch.gather(u_values, 0, u_sorter)

    v_sorter = torch.sort(v_values, 0)[1]
    v_values = torch.gather(v_values, 0, v_sorter)

    u_weights = torch.gather(u_weights, 0, u_sorter)
    v_weights = torch.gather(v_weights, 0, v_sorter)

    u_cumweights = torch.cumsum(u_weights, 0)
    v_cumweights = torch.cumsum(v_weights, 0)

    qs = torch.sort(torch.cat((u_cumweights, v_cumweights), 0), 0)[0]
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)

    pad_width = [(1, 0)] + (qs.ndim - 1) * [(0, 0)]
    how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
    qs = pad(qs, how_pad)

    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = torch.abs(u_quantiles - v_quantiles)
    
    return torch.mean(torch.sum(delta * torch.pow(diff_quantiles, p), dim=0, keepdim=True))**(1./p)

def sliced_Wasserstein(X, Y, a, b, L, p=2):
    dx = X.shape[1]
    dy = Y.shape[1]
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    thetas_x = torch.randn(dx, L, device=X.device)
    thetas_x = thetas_x/torch.sqrt(torch.sum(thetas_x**2, dim=0, keepdim=True))

    thetas_y = torch.randn(dy, L, device=X.device)
    thetas_y = thetas_y/torch.sqrt(torch.sum(thetas_y**2, dim=0, keepdim=True))

    X_prod = torch.matmul(X, thetas_x)
    Y_prod = torch.matmul(Y, thetas_y)
    return one_dimensional_Wasserstein(X_prod, Y_prod, a, b, p)

def l2_prj_dist(X, Y, L=1000):
    d = X.shape[1]
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    thetas = torch.randn(d, L, device=X.device)
    thetas = thetas/torch.sqrt(torch.sum(thetas**2, dim=0, keepdim=True))
    X_prod = torch.matmul(X, thetas)
    Y_prod = torch.matmul(Y, thetas)
    return torch.cdist(X_prod, Y_prod, p=2)


# def rand_projections(dim, num_projections=1000,device='cpu'):
#     projections = torch.randn((num_projections, dim),device=device)
#     projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
#     return projections


# def one_dimensional_Wasserstein_prod(X,Y,theta,p):
#     X_prod = torch.matmul(X, theta.transpose(0, 1))
#     Y_prod = torch.matmul(Y, theta.transpose(0, 1))
#     X_prod = X_prod.view(X_prod.shape[0], -1)
#     Y_prod = Y_prod.view(Y_prod.shape[0], -1)
#     wasserstein_distance = torch.abs(
#         (
#                 torch.sort(X_prod, dim=0)[0]
#                 - torch.sort(Y_prod, dim=0)[0]
#         )
#     )
#     wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
#     return wasserstein_distance

# def SW(X, Y, L=10, p=2, device="cpu"):
#     dim = X.size(1)
#     theta = rand_projections(dim, L,device)
#     sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
#     return  torch.pow(sw,1./p)

if __name__ =="__main__":
    # list_prj = [init_prj_matrix(128) for i in range(10)]
    x = torch.tensor([3.2, 3.1, 2.4, 2.5]).view(4,1)
    y = torch.tensor([13.0,40.1, 10.0, 20.0]).view(4,1)

    dist = sliced_Wasserstein(x,y,a=None, b=None, L=1)

    # dist = euclidean_prj_dist(list_prj, a, b)
    print(dist)