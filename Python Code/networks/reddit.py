# This code is from Updated Testing Reddit - No Con- bias (Fictitious Play)-01092022
# This code replace the big real datanetwork with small sythetic network
import os
import scipy.io
import scipy
import numpy as np
import networkx as nx
import sys

# add parent directory to path so that utils is available
sys.path.append('..')
from utils import num_edges, get_gap


def init():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../data/Reddit.mat")

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


def network_anl(s, n, G, agent):
    print(f'{str(agent)} opinion: {str(s[agent])}')
    print(f'{str(agent)} neighbors: {str(np.nonzero(G[agent]))}')

    s_aa = s[:, 0]
    my_dict = {index: value for index, value in enumerate(s_aa)}
    sorting_s = sorted(my_dict.items(), key=lambda x: x[1])
    sorted_S = dict(sorting_s)

    temp = list(sorted_S.items())
    res = [idx for idx, key in enumerate(temp) if key[0] == agent]
    print(f"Opinion rank of this agent is: {str(res)}")

    # print("___________________Max Analyze__________________________________________")
    nxG = nx.from_numpy_array(G)
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
    gap = get_gap(s, n)
    my_gap = {index: value for index, value in enumerate(gap)}
    sorting_gap = sorted(my_gap.items(), key=lambda x: x[1])
    sorted_gap = dict(sorting_gap)
    # print(sorted_gap)
    temp4 = list(sorted_gap.items())
    res4 = [idx for idx, key in enumerate(temp4) if key[0] == agent]
    print("Agent's opinion gap to mean opinion is ranked as: " + str(res4))