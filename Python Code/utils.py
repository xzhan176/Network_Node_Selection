import numpy as np
import pandas as pd

### Utility functions ###


def rank(scores, agent):
    ranks = 1
    for i in scores:
        if scores[agent] > scores[i]:
            ranks += 1
        elif scores[agent] == scores[i]:
            ranks = ranks
    return ranks


def mean_center(op, n):
    """centers the opinion vector around 0"""
    ones = np.ones((n, 1))
    x = op - (np.dot(np.transpose(op), ones)/n) * ones
    return x


def num_edges(L, n):
    """computes the number of edges in the network"""
    m = 0
    for i in range(n):
        for j in range(n):
            if i > j and L[i, j] < 0:
                m += 1
    return m


def get_node_edges(G, n):
    edges = []
    for v in range(n):
        a = np.array(np.nonzero(G[v])[0])
        edge = len(a)
#         print(edge)
        edges.append(edge)

    return edges


def obj_polarization(A, L, op, n):
    # maximizing polarization only: \\bar{z}^T \\bar{z}
    op_mean = mean_center(op, n)
    z_mean = np.dot(A, op_mean)
    return np.dot(np.transpose(z_mean), z_mean)[0, 0]

# def obj_polarization_1(A, L, op, n):  #z_mean is the same as s_mean - according to Stanford paper theory
#     z = np.dot(A, op)
#     z_mean = mean_center(z, n)
#     return np.dot(np.transpose(z_mean), z_mean)[0,0]

# TODO unused
# def obj_innate_polarization(s, n):
#     """Calculate innate polarization"""
#     #     np.set_printoptions(precision=5)
#     op_mean = mean_center(s, n)
#     return np.dot(np.transpose(op_mean), op_mean)[0, 0]


def calculate_centrality_and_convert_to_df(network, centrality_func):
    centrality = centrality_func(network)
    df = pd.DataFrame(list(centrality.values()))
    return df
