# This code is from Updated Testing Reddit - No Con- bias (Fictitious Play)-01092022
# This code replace the big real datanetwork with small sythetic network
import os
import scipy.io
import scipy
import numpy as np


def init():
    # Load the Reddit.mat file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data/Reddit.mat")
    data = scipy.io.loadmat(file_path)

    # number of vertices = 556
    n = data['Reddit'][0, 0][0].shape[0]
    # adjacency matrix in compressed sparse column format, convert to array
    G = data['Reddit'][0, 0][0].toarray()
    # labeled "recent innate opinions"
    s = data['Reddit'][0, 0][5]

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

    return G, s, n
