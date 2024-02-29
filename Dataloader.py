"""
@Project   : IMvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : Dataloader.py
"""

import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize


def load_data(args, device):
    data = sio.loadmat(args.path + args.dataset + ".mat")
    features = data["X"]
    feature_list = []
    flt_list = []

    labels = data["Y"].flatten()
    labels = labels - min(set(labels))

    # 所以n-repeat没有随机样本，只是重复实验
    idx_labeled = data["train" + str(args.ratio)].squeeze(0).tolist()
    idx_unlabeled = data["test" + str(args.ratio)].squeeze(0).tolist()
    labels = torch.from_numpy(labels).long()

    flt_f = torch.zeros(features[0][0].shape[0], features[0][0].shape[0])
    for i in range(features.shape[1]):  # 形状应该是[[v1, v2, ...]]
        features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix(feature):  # sparse矩阵和dense矩阵存储方式不同
            feature = feature.todense()
        direction = "./data/lp_matrix/" + args.dataset + "/" + "v" + str(i) + ".npz"
        print(
            "Loading laplacian matrix from the " + str(i) + "th view of " + args.dataset
        )
        lp = torch.from_numpy(ss.load_npz(direction).todense()).float()

        flt = torch.eye(lp.shape[0]) - args.Lambda * lp
        flt_f += flt / features.shape[1]
        flt = construct_sparse_float_tensor(flt).to(device)
        feature = torch.from_numpy(feature).float().to(device)
        # feature = construct_sparse_float_tensor(torch.from_numpy(feature)).to(device)
        feature_list.append(feature)
        # TODO 所以说图卷积不需要在每层卷积之前重新构建图拉普拉斯，用的都是开始的卷积核(L)
        flt_list.append(flt)
        del feature, flt, lp
        torch.cuda.empty_cache()
    flt_f_ = construct_sparse_float_tensor(flt_f).to(device)
    del flt_f
    torch.cuda.empty_cache()

    return feature_list, flt_list, flt_f_, labels, idx_labeled, idx_unlabeled


def construct_sparse_float_tensor(np_matrix):
    """
        construct a sparse float tensor according a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
    sp_matrix = ss.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(
        torch.LongTensor(three_tuple[0].T),
        torch.FloatTensor(three_tuple[1]),
        torch.Size(three_tuple[2]),
    )
    return sparse_tensor


def sparse_to_tuple(sparse_mx):
    if not ss.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    # sparse_mx.row/sparse_mx.col  <class 'numpy.ndarray'> [   0    0    0 ... 2687 2694 2706]
    coords = np.vstack(
        (sparse_mx.row, sparse_mx.col)
    ).transpose()  # <class 'numpy.ndarray'> (n_edges, 2)
    values = sparse_mx.data  # <class 'numpy.ndarray'> (n_edges,) [1 1 1 ... 1 1 1]
    shape = sparse_mx.shape  # <class 'tuple'>  (n_samples, n_samples)
    return coords, values, shape
