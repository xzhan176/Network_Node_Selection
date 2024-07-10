import numpy as np
import pandas as pd
import importlib
import networkx as nx


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
        edges.append(edge)

    return edges


def obj_polarization(A, op, n):
    """
    Maximizing polarization only: \\bar{z}^T \\bar{z}
    """
    op_mean = mean_center(op, n)
    z_mean = np.dot(A, op_mean)
    return np.dot(np.transpose(z_mean), z_mean)[0, 0]


def obj_innate_polarization(s, n):
    """
    Calculate innate polarization
    """
    # np.set_printoptions(precision=5)
    op_mean = mean_center(s, n)
    return np.dot(np.transpose(op_mean), op_mean)[0, 0]


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


def calculate_polarization(s, n, A, L):
    y = mean_center(s, n)
    # Polarization before opinion dynamics
    innat_pol = np.dot(np.transpose(y), y)[0, 0]
    print(f'Innate_polarization:\t{innat_pol}')

    # Polarization after opinion dynamics
    equ_pol = obj_polarization(A, s, n)
    print(f'Equi_polarization:\t{equ_pol}')

    di = equ_pol - innat_pol
    print(f"Difference:\t\t{di}")


def import_network(name: str):
    return importlib.import_module(f'networks.{name}')


def network_anl(s, n, G, agent):
    print(f'{agent} opinion: {s[agent]}')
    print(f'{agent} neighbors: {np.nonzero(G[agent])}')

    s_aa = s[:, 0]
    my_dict = {index: value for index, value in enumerate(s_aa)}
    sorting_s = sorted(my_dict.items(), key=lambda x: x[1])

    sorted_S = dict(sorting_s)
    res = rank(sorted_S, agent)
    print(f"Opinion rank of this agent is: {res}")

    nxG = nx.from_numpy_array(G)
    print("_______________Degree Centrality___________________")
    deg_centrality = nx.degree_centrality(nxG)
    sortedDict = sorted(deg_centrality.items(), key=lambda x: x[1])
    converted_dict = dict(sortedDict)
    res1 = rank(converted_dict, agent)+1
    print(f"rank of this agent is:\t{res1}")
    print(converted_dict[agent])
    # print(converted_dict)

    print("\n_______________Closeness Rank________________________")
    close_centrality = nx.closeness_centrality(nxG)
    sortedDict1 = sorted(close_centrality.items(), key=lambda x: x[1])
    converted_dict1 = dict(sortedDict1)
    res2 = rank(converted_dict1, agent)+1
    print(f"rank of this agent is:\t{res2}")
    print(converted_dict1[agent])
    # print(converted_dict1)

    print("\n_______________Page Rank_____________________________")
    pr = nx.eigenvector_centrality(nxG)
    sortedDict3 = sorted(pr.items(), key=lambda x: x[1])
    converted_dict3 = dict(sortedDict3)
    res3 = rank(converted_dict3, agent)+1
    print(f"rank of this agent is:\t{str(res3)}")
    print(converted_dict3[agent])
    # print(converted_dict3)

    gaps = get_gap(s, n)
    if gaps[agent] < 0:
        my_gap = {index: value for (index, value)
                  in enumerate(gaps) if value < 0}
        sorting_gap = sorted(my_gap.items(), key=lambda x: x[1])
        sorted_gap = dict(sorting_gap)
        res4 = rank(sorted_gap, agent)
    else:
        my_gap = {index: value for index,
                  value in enumerate(gaps) if value >= 0}
        sorting_gap = sorted(my_gap.items(), key=lambda x: x[1], reverse=True)
        sorted_gap = dict(sorting_gap)
        res4 = rank(sorted_gap, agent)

    print(f"Agent's opinion extremity is ranked as:\t{res4}")
    print(f"Agent's min_pref is ranked as:\t{res4+res1}")

# def exportGameResult(game: Game, result: GameResult, k, experiment):
#     pd.DataFrame(result.payoff_matrix).to_csv(
#         f'results/Payoff-Matrix-k-{k}-experiment-{experiment}.csv')

#     # Save the original standard output
#     original_stdout = sys.stdout

#     with open(f'results/Result-k-{k}-experiment-{experiment}.txt', "w") as f:
#         # Change the standard output to the file we created.
#         sys.stdout = f

#         print('Initial Condition -(agent, opinion, pol)')
#         # print(f'Innate op {s}')
#         # print(f'Adjacency matrix {G}')
#         # print('Selected Nodeset, k_Opinions, Steady-state polarization')
#         print(f'Max:\t{result.first_max}')
#         print(f'Min:\t{result.first_min}')

#         print('_____________________')
#         print(f'Max Pol:\t{result.equi_max}')
#         print(f'Min Pol:\t{result.equi_min}')

#         # MAXimizer's distribution of LAST 100 iteration
#         print('Max_distribution_last_100')
#         max_l100_fre = result.max_history_last_100/100
#         print(max_l100_fre[np.nonzero(max_l100_fre)])
#         # print for small network
#         # print(max_history_last_100)
#         # # Print for Large Network
#         print(np.nonzero(max_l100_fre))

#         columns = np.nonzero(max_l100_fre)
#         columns = list(columns[0])
#         for column in columns:
#             (k_nodes, opinions) = game.map_action(column)
#             print(f'Max Nodes: {k_nodes}\tOpinion: {opinions}')

#         print('Max_distribution_all')
#         max_fre = result.max_history/result.game_rounds
#         print(max_fre[np.nonzero(max_fre)])
#         print([np.nonzero(max_fre)])

#         # TODO confirm isn't this block the same as the one above
#         columns_all = np.nonzero(max_l100_fre)
#         columns_all = list(columns_all[0])
#         for column in columns_all:
#             (k_nodes, opinions) = game.map_action(column)
#             print(f'Max Nodes: {k_nodes}\tOpinion: {opinions}')

#         # MINimizer's Strategy in the last 100 round
#         counter = collections.Counter(result.min_touched_last_100)
#         # frequency of all min options in order
#         fla_min_fre = np.array(list(counter.values()))/(100)
#         print('Min_distribution_last_100')
#         print(fla_min_fre)
#         print(counter)
#         # print(min_touched_last_100)

#         # a dictionary include {'min_option': count of this choice}
#         counter_1 = collections.Counter(result.min_touched_all)
#         # frequency of all min options in order
#         fla_min_fre_1 = np.array(list(counter_1.values()))/result.game_rounds
#         print('Min_distribution_all')
#         print(fla_min_fre_1)
#         print(counter_1)
#         np.set_printoptions(precision=3)

#         # a dictionary include {'min_option': count of this choice}
#         counter_a = collections.Counter(result.min_history)
#         print(counter_a)

#         print(f'min_recent_{game.memory}_touched')
#         print(result.min_touched)
#         print(f'max_recent_{game.memory}_touched')
#         print(result.max_touched)

#         # Reset the standard output to its original value
#         sys.stdout = original_stdout