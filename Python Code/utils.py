import numpy as np
import pandas as pd
import importlib
from itertools import product, combinations


def rank(scores, agent):
    ranks = 1
    for i in scores:
        if scores[agent] > scores[i]:
            ranks += 1
        elif scores[agent] == scores[i]:
            ranks = ranks
    return ranks


def mean_center(op, n):
    """
    Centers the opinion vector around 0
    """
    ones = np.ones((n, 1))
    x = op - (np.dot(np.transpose(op), ones)/n) * ones
    return x


def num_edges(L, n):
    """
    Computes the number of edges in the network
    """
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
    """
    Maximizing polarization only: \\bar{z}^T \\bar{z}
    """
    op_mean = mean_center(op, n)
    z_mean = np.dot(A, op_mean)
    return np.dot(np.transpose(z_mean), z_mean)[0, 0]

# def obj_polarization_1(A, L, op, n):  #z_mean is the same as s_mean - according to Stanford paper theory
#     z = np.dot(A, op)
#     z_mean = mean_center(z, n)
#     return np.dot(np.transpose(z_mean), z_mean)[0,0]


def obj_innate_polarization(s, n):
    """
    Calculate innate polarization
    """
    # np.set_printoptions(precision=5)
    op_mean = mean_center(s, n)
    return np.dot(np.transpose(op_mean), op_mean)[0, 0]


def len_actions(k, n):
    """
    Calculate the length of all possible actions for k opinions and n nodes
    """
    # create all combination of K opinions
    max_option = [0, 1]
    k_opinions = list(product(max_option, repeat=k))
    # Horizontal length of all possible actions
    all = list(range(n))
    # all possible actions = all combination of k nodes * all combination of k opinions
    h = len(list(combinations(all, k))) * len(k_opinions)
    return h


def calculate_centrality_and_convert_to_df(network, centrality_func):
    centrality = centrality_func(network)
    df = pd.DataFrame(list(centrality.values()))
    return df


def get_gap(op, n):
    ones = np.ones((n, 1))
    x = op - (np.dot(np.transpose(op), ones)/n) * ones
    return x


def plot_centrality_histogram(ax, df, title, bins, ylim):
    ax.hist(df, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel('Number of Nodes', fontsize=16)
    ax.set_ylim(0, ylim)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)


def calculate_polarization1(s, n, A, L):
    y = mean_center(s, n)
    # Polarization before opinion dynamics
    innat_pol = np.dot(np.transpose(y), y)[0, 0]
    print(f'Innate_polarization:\t{innat_pol}')

    # Polarization after opinion dynamics
    equ_pol = obj_polarization(A, L, s, n)
    print(f'Equi_polarization:\t{equ_pol}')

    di = equ_pol - innat_pol
    print(f"Difference:\t\t{di}")


def import_network(name):
    return importlib.import_module(f'networks.{name}')
