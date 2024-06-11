import networkx as nx
import numpy as np
import pandas as pd
import sys

# add parent directory to path so that utils is available
sys.path.append('..')
from utils import rank, get_gap


def init():
    """
    Generates a Karate Club graph, creates an adjacency matrix, and generates innate opinions.

    Returns:
    - G: The Karate Club graph as a numpy array.
    - s: The innate opinions as a numpy array.
    - n: The number of agents in the graph.
    """

    G = nx.karate_club_graph()
    edges = []
    n = 0

    for v in G:
        a = f"{G.degree(v):6}"
        edges.append(a)
        n += 1
    print(f'There are {n} agents')

    ############################ Make Adjacency Matrix #####################################
    ZKC_graph = nx.karate_club_graph()
    G = nx.convert_matrix.to_numpy_array(ZKC_graph)
    G[G != 0] = 1
    print(G)

    ############################ Make Innate Opinion ################################
    # create two set of weights connected with density 1) individuals  2) individual & information Source
    # assuming (1-r) are individuals
    c1 = np.sort(np.random.choice(n, n, replace=False))
    l1 = len(c1)

    # Import Innate opinion
    df = pd.read_csv('data/Karate Innate Opinion.csv')
    s_1 = np.array(df[df.columns[1]])

    s = np.reshape(s_1, (34, -1))
    # print(s)
    return G, s, n


def network_anl(s, n, G, agent):
    print(f'{agent} opinion: {s[agent]}')
    print(f'{agent} neighbors: {np.nonzero(G[agent])}')

    s_aa = s[:, 0]
    my_dict = {index: value for index, value in enumerate(s_aa)}
    sorting_s = sorted(my_dict.items(), key=lambda x: x[1])

    sorted_S = dict(sorting_s)
    res = rank(sorted_S, agent)
    print(f"Opinion rank of this agent is: {res}")

    # print("___________________Max Analyze__________________________________________")
    nxG = nx.from_numpy_array(G)
    print("_______________Degree Centrality___________________")
    deg_centrality = nx.degree_centrality(nxG)
    sortedDict = sorted(deg_centrality.items(), key=lambda x: x[1])
    converted_dict = dict(sortedDict)
    res1 = rank(converted_dict, agent)+1
    print(f"rank of this agent is:\t{res1}")
    print(converted_dict[agent])
    # print(converted_dict)

    print("                           ")
    print("_______________Closeness Rank________________________")
    close_centrality = nx.closeness_centrality(nxG)
    sortedDict1 = sorted(close_centrality.items(), key=lambda x: x[1])
    converted_dict1 = dict(sortedDict1)
    res2 = rank(converted_dict1, agent)+1
    print(f"rank of this agent is:\t{res2}")
    print(converted_dict1[agent])
    # print(converted_dict1)

    print("                           ")
    print("_______________Page Rank_____________________________")
    pr = nx.eigenvector_centrality(nxG)
    sortedDict3 = sorted(pr.items(), key=lambda x: x[1])
    converted_dict3 = dict(sortedDict3)
    res3 = rank(converted_dict3, agent)+1
    print(f"rank of this agent is:\t{str(res3)}")
    print(converted_dict3[agent])
    # print(converted_dict3)

    print("                           ")
    gaps = get_gap(s, n)
    if gaps[agent] < 0:
        my_gap = {index: value for index, value in enumerate(gaps) if value < 0}
        sorting_gap = sorted(my_gap.items(), key=lambda x: x[1])
        sorted_gap = dict(sorting_gap)
        res4 = rank(sorted_gap, agent)
        # temp4 = list(sorted_gap.items())
        # res4 = [idx for idx, key in enumerate(temp4) if key[0]==agent][0]+1
    else:
        my_gap = {index: value for index, value in enumerate(gaps) if value >= 0}
        sorting_gap = sorted(my_gap.items(), key=lambda x: x[1], reverse=True)
        sorted_gap = dict(sorting_gap)
        res4 = rank(sorted_gap, agent)
        # temp4 = list(sorted_gap.items())
        # res4 = [idx for idx, key in enumerate(temp4) if key[0]==agent][0]+1
    print(f"Agent's opinion extremity is ranked as:\t{res4}")
    print(f"Agent's min_pref is ranked as:\t{res4+res1}")
