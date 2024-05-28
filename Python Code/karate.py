import networkx as nx
import numpy as np
import pandas as pd


def karate():
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
        n = n + 1
    print('There are ' + str(n) + ' agents')

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


def rank(scores, agent):
    ranks = 1
    for i in scores:
        if scores[agent] > scores[i]:
            ranks += 1
        elif scores[agent] == scores[i]:
            ranks = ranks
    return ranks


def network_anl(s, n, G, agent):

    print(str(agent)+' opinion: ' + str(s[agent]))
    print(str(agent)+' neighbors: ' + str(np.nonzero(G[agent])))

    s_aa = s[:, 0]
    my_dict = {index: value for index, value in enumerate(s_aa)}
    sorting_s = sorted(my_dict.items(), key=lambda x: x[1])
    sorted_S = dict(sorting_s)
    res = rank(sorted_S, agent)
    # printing result
    print("Opinion rank of this agent is : " + str(res))

    # print("___________________Max Analyze__________________________________________")
    nxG = nx.from_numpy_matrix(G)
    # G = nx.karate_club_graph()
    print("_______________Degree Centrality___________________")
    deg_centrality = nx.degree_centrality(nxG)
    sortedDict = sorted(deg_centrality.items(), key=lambda x: x[1])
    converted_dict = dict(sortedDict)
    res1 = rank(converted_dict, agent)+1
    print("rank of this agent is : " + str(res1))
    print(converted_dict[agent])

    # print(converted_dict)
    print("                           ")
    print("_______________Closeness Rank________________________")
    close_centrality = nx.closeness_centrality(nxG)
    sortedDict1 = sorted(close_centrality.items(), key=lambda x: x[1])
    converted_dict1 = dict(sortedDict1)
    res2 = rank(converted_dict1, agent)+1
    print("rank of this agent is : " + str(res2))
    print(converted_dict1[agent])
    # print(converted_dict1)
    print("                           ")
    print("_______________Page Rank_____________________________")
    pr = nx.eigenvector_centrality(nxG)
    sortedDict3 = sorted(pr.items(), key=lambda x: x[1])
    converted_dict3 = dict(sortedDict3)
    res3 = rank(converted_dict3, agent)+1
    print("rank of this agent is : " + str(res3))
    print(converted_dict3[agent])
    # print(converted_dict3)

    print("                           ")

    def gap(op, n):
        ones = np.ones((n, 1))
        x = op - (np.dot(np.transpose(op), ones)/n) * ones
        return x

    gaps = gap(s, n)
    if gaps[agent] < 0:
        my_gap = {index: value for index,
                  value in enumerate(gaps) if value < 0}
        sorting_gap = sorted(my_gap.items(), key=lambda x: x[1])
        sorted_gap = dict(sorting_gap)
        res4 = rank(sorted_gap, agent)
#         temp4 = list(sorted_gap.items())
#         res4 = [idx for idx, key in enumerate(temp4) if key[0]==agent][0]+1
    else:
        my_gap = {index: value for index,
                  value in enumerate(gaps) if value >= 0}
        sorting_gap = sorted(my_gap.items(), key=lambda x: x[1], reverse=True)
        sorted_gap = dict(sorting_gap)
        res4 = rank(sorted_gap, agent)
#         temp4 = list(sorted_gap.items())
#         res4 = [idx for idx, key in enumerate(temp4) if key[0]==agent][0]+1
    print("Agent's opinion extremity is ranked as: " + str(res4))
    print("Agent's min_pref is ranked as: " + str(res4+res1))
