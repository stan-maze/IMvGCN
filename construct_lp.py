"""
@Project   : IMvGCN
@Time      : 2023/4/10
@Author    : Zhihao Wu
@File      : construct_lp.py
"""
import time
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import argparse


def construct_lp_matrix(fea, knn):
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=knn + 1, algorithm='ball_tree').fit(fea)
    adj_construct = nbrs.kneighbors_graph(fea)  # <class 'scipy.sparse.csr.csr_matrix'>
    adj = ss.coo_matrix(adj_construct)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_ = ss.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    lp = ss.eye(adj.shape[0]) - adj_wave
    print("Time cost: ", time.time() - start_time)
    return lp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./data/datasets/", help="Dataset path")
    parser.add_argument("--dataset", type=str, default="Citeseer", help="Dataset name")
    parser.add_argument("--knn", type=int, default=30, help="k nearest neighbors")
    args = parser.parse_args()
    # 读入的可以是原始的数据，例如原本的MNIST(60000, 780)
    # 也可以是通过提取器提取出来的特征如通过
    #   30-D IsoProjection, 9-D Linear Discriminant Analysis and 9-D Neighborhood Preserving Embedding features
    #   提取出来的multi-view MNIST((2000, 30),(2000, 9),(2000, 30))
    data = sio.loadmat(args.path + args.dataset + '.mat')
    features = data['X']
    
    print(dir(data))
    print(data.keys())
    print(data['__header__'])
    
    # features = data['Z']
    print(features.shape)
    print(features)
    # exit(0)

    for i in range(features.shape[1]):
        feature = normalize(features[0][i])
        print(feature.shape)
        continue
        if ss.isspmatrix(feature):
            feature = feature.todense()
        print("Constructing the laplacian matrix of " + str(i) + "th view of " + args.dataset)
        lp = construct_lp_matrix(feature, args.knn)
        save_direction = './data/lp_matrix/' + args.dataset + '/' + 'v' + str(i) + '.npz'
        ss.save_npz(save_direction, lp)
