import argparse
import scipy
from game import Game, exportGameResult
from utils import *


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("network", help="The network to run the experiment on")
    parser.add_argument(
        "rounds", help="The number of rounds to run the game for", type=int)
    parser.add_argument("memory", help="The memory of the game", type=int)
    args = parser.parse_args()

    print(f"Running experiment for network \"{args.network}\"")

    # Import network
    network = import_network(args.network)
    G, s, n = network.init()
    L = scipy.sparse.csgraph.laplacian(G, normed=False)
    A = np.linalg.inv(np.identity(n) + L)

    # Configure the game
    game_rounds = 3
    memory = 10
    k = 2

    # Run the game
    game = Game(s, A, L, k)
    result = game.run(game_rounds, memory)

    # Save the result
    experiment = 1
    exportGameResult(game, result, k, memory, experiment)


if __name__ == "__main__":
    main()
