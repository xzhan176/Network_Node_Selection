import networkx as nx
import numpy as np
import pandas as pd
import os


def init():
    """
    Generates a Karate Club graph, creates an adjacency matrix, and generates innate opinions.

    Returns:
    - G: The Karate Club graph as a numpy array.
    - s: The innate opinions as a numpy array.
    - n: The number of agents in the graph.
    """

    G = nx.karate_club_graph()
    n = len(G)

    print(f'There are {n} agents')

    ############################ Make Adjacency Matrix #####################################
    ZKC_graph = nx.karate_club_graph()
    G = nx.convert_matrix.to_numpy_array(ZKC_graph)
    G[G != 0] = 1
    print(G)

    # Import Innate opinion
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/Karate Innate Opinion.csv")
    df = pd.read_csv(data_path)
    s_1 = np.array(df[df.columns[1]])

    s = np.reshape(s_1, (34, -1))
    return G, s, n
