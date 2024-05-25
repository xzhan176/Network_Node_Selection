# This code is from Updated Testing Reddit - No Con- bias (Fictitious Play)-01092022
# This code replace the big real datanetwork with small sythetic network
import os
import scipy.io
import scipy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import beta
# %run pure_strategy_selection.ipynb  #include simple selection algorithm

# centers the opinion vector around 0\n",


def mean_center(op, n):
    ones = np.ones((n, 1))
    x = op - (np.dot(np.transpose(op), ones)/n) * ones
    return x

# compute number of edges, m\n


def num_edges(L, n):
    m = 0
    for i in range(n):
        for j in range(n):
            if i > j and L[i, j] < 0:
                m += 1
    return m

# maximizing polarization only: \\bar{z}^T \\bar{z}


def obj_polarization(A, L, op, n):
    op_mean = mean_center(op, n)
    z_mean = np.dot(A, op_mean)
    return np.dot(np.transpose(z_mean), z_mean)[0, 0]

# def obj_polarization_1(A, L, op, n):  #z_mean is the same as s_mean - according to Stanford paper theory
#     z = np.dot(A, op)
#     z_mean = mean_center(z, n)
#     return np.dot(np.transpose(z_mean), z_mean)[0,0]

# Calculate innate polarization


def obj_innate_polarization(s, n):
    #     np.set_printoptions(precision=5)
    op_mean = mean_center(s, n)
    return np.dot(np.transpose(op_mean), op_mean)[0, 0]


def network_anl(s, n, G, agent):

    print(str(agent)+' opinion: ' + str(s[agent]))
    print(str(agent)+' neighbors: ' + str(np.nonzero(G[agent])))

    s_aa = s[:, 0]
    my_dict = {index: value for index, value in enumerate(s_aa)}
    sorting_s = sorted(my_dict.items(), key=lambda x: x[1])
    sorted_S = dict(sorting_s)

    temp = list(sorted_S.items())
    res = [idx for idx, key in enumerate(temp) if key[0] == agent]
    # printing result
    print("Opinion rank of this agent is : " + str(res))

    # print("___________________Max Analyze__________________________________________")
    nxG = nx.from_numpy_matrix(G)
    # G = nx.karate_club_graph()
    print("_______________Degree Centrality___________________")
    deg_centrality = nx.degree_centrality(nxG)
    sortedDict = sorted(deg_centrality.items(), key=lambda x: x[1])
    converted_dict = dict(sortedDict)
    temp1 = list(converted_dict.items())
    res1 = [idx for idx, key in enumerate(temp1) if key[0] == agent]
    print("rank of this agent is : " + str(res1))
    print(converted_dict[agent])

    # print(converted_dict)
    print("                           ")
    print("_______________Closeness Rank________________________")
    close_centrality = nx.closeness_centrality(nxG)
    sortedDict1 = sorted(close_centrality.items(), key=lambda x: x[1])
    converted_dict1 = dict(sortedDict1)
    temp2 = list(converted_dict1.items())
    res2 = [idx for idx, key in enumerate(temp2) if key[0] == agent]
    print("rank of this agent is : " + str(res2))
    print(converted_dict1[agent])
    # print(converted_dict1)
    print("                           ")
    print("_______________Page Rank_____________________________")
    pr = nx.eigenvector_centrality(nxG)
    sortedDict3 = sorted(pr.items(), key=lambda x: x[1])
    converted_dict3 = dict(sortedDict3)
    temp3 = list(converted_dict3.items())
    res3 = [idx for idx, key in enumerate(temp3) if key[0] == agent]
    print("rank of this agent is : " + str(res3))
    print(converted_dict3[agent])
    # print(converted_dict3)

    print("                           ")

    def gap(op, n):
        ones = np.ones((n, 1))
        x = op - (np.dot(np.transpose(op), ones)/n) * ones
        return abs(x)

    gap = gap(s, n)
    my_gap = {index: value for index, value in enumerate(gap)}
    sorting_gap = sorted(my_gap.items(), key=lambda x: x[1])
    sorted_gap = dict(sorting_gap)
    # print(sorted_gap)
    temp4 = list(sorted_gap.items())
    res4 = [idx for idx, key in enumerate(temp4) if key[0] == agent]
    print("Agent's opinion gap to mean opinion is ranked as: " + str(res4))


def network_anl(s, n, G, agent):

    print(str(agent)+' opinion: ' + str(s[agent]))
    print(str(agent)+' neighbors: ' + str(np.nonzero(G[agent])))

    s_aa = s[:, 0]
    my_dict = {index: value for index, value in enumerate(s_aa)}
    sorting_s = sorted(my_dict.items(), key=lambda x: x[1])
    sorted_S = dict(sorting_s)

    temp = list(sorted_S.items())
    res = [idx for idx, key in enumerate(temp) if key[0] == agent]
    # printing result
    print("Opinion rank of this agent is : " + str(res))

    # print("___________________Max Analyze__________________________________________")
    nxG = nx.from_numpy_matrix(G)
    # G = nx.karate_club_graph()
    print("_______________Degree Centrality___________________")
    deg_centrality = nx.degree_centrality(nxG)
    sortedDict = sorted(deg_centrality.items(), key=lambda x: x[1])
    converted_dict = dict(sortedDict)
    temp1 = list(converted_dict.items())
    res1 = [idx for idx, key in enumerate(temp1) if key[0] == agent]
    print("rank of this agent is : " + str(res1))
    print(converted_dict[agent])

    # print(converted_dict)
    print("                           ")
    print("_______________Closeness Rank________________________")
    close_centrality = nx.closeness_centrality(nxG)
    sortedDict1 = sorted(close_centrality.items(), key=lambda x: x[1])
    converted_dict1 = dict(sortedDict1)
    temp2 = list(converted_dict1.items())
    res2 = [idx for idx, key in enumerate(temp2) if key[0] == agent]
    print("rank of this agent is : " + str(res2))
    print(converted_dict1[agent])
    # print(converted_dict1)
    print("                           ")
    print("_______________Page Rank_____________________________")
    pr = nx.eigenvector_centrality(nxG)
    sortedDict3 = sorted(pr.items(), key=lambda x: x[1])
    converted_dict3 = dict(sortedDict3)
    temp3 = list(converted_dict3.items())
    res3 = [idx for idx, key in enumerate(temp3) if key[0] == agent]
    print("rank of this agent is : " + str(res3))
    print(converted_dict3[agent])
    # print(converted_dict3)

    print("                           ")

    def gap(op, n):
        ones = np.ones((n, 1))
        x = op - (np.dot(np.transpose(op), ones)/n) * ones
        return abs(x)

    gap = gap(s, n)
    my_gap = {index: value for index, value in enumerate(gap)}
    sorting_gap = sorted(my_gap.items(), key=lambda x: x[1])
    sorted_gap = dict(sorting_gap)
    # print(sorted_gap)
    temp4 = list(sorted_gap.items())
    res4 = [idx for idx, key in enumerate(temp4) if key[0] == agent]
    print("Agent's opinion gap to mean opinion is ranked as: " + str(res4))


def reddit():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data/Reddit.mat")

    # # Define the file path
    # file_path = r"/Users/es2330/Downloads/12092023_PoGame/data/Reddit.mat"

    # Load the Reddit.mat file
    data = scipy.io.loadmat(file_path)
    n = data['Reddit'][0, 0][0].shape[0]     # number of vertices = 556
    # adjacency matrix in compressed sparse column format, convert to array
    G = data['Reddit'][0, 0][0].toarray()
    # mapping from node ID to labels 1-556 (not important)
    nodemap = data['Reddit'][0, 0][1]
    edges = data['Reddit'][0, 0][2]     # list of edges (same as G, not used)
    s = data['Reddit'][0, 0][5]     # labeled "recent innate opinions"

    # remove isolated vertices from the graph
    s = np.delete(s, 551)
    s = np.delete(s, 105)
    s = np.delete(s, 52)
    n -= 3
    s = s.reshape((n, 1))
    G = np.delete(G, 551, 1)
    G = np.delete(G, 551, 0)
    G = np.delete(G, 105, 1)
    G = np.delete(G, 105, 0)
    G = np.delete(G, 52, 1)
    G = np.delete(G, 52, 0)

    L = scipy.sparse.csgraph.laplacian(
        G, normed=False)  # Return the Laplacian matrix
    # A = (I + L)^(-1)\n  Stanford paper theory
    A = np.linalg.inv(np.identity(n) + L)
    # call the function to calculate the number of edges
    m = num_edges(L, n)
    columnsum_ij = np.sum(A, axis=0)
    # print(columnsum_ij)
    return G, s, n
