import argparse
import scipy
from game import Game, exportGameResult
from utils import *


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("network",
                        help="The network to run the experiment on")
    parser.add_argument("k",
                        help="The number of nodes to select",
                        type=int)
    parser.add_argument("experiment",
                        help="The experiment number to run",
                        type=int)
    parser.add_argument("-r", "--rounds",
                        help="The number of rounds to run the game for. Default is k * 200",
                        type=int)
    parser.add_argument("-m", "--memory",
                        help="The memory of the game. Default is 10",
                        type=int,
                        default=10)
    args = parser.parse_args()

    # Import network
    network_module = import_network(args.network)
    G, s, n = network_module.init()
    L = scipy.sparse.csgraph.laplacian(G, normed=False)
    A = np.linalg.inv(np.identity(n) + L)

    # Configure the game
    memory = args.memory
    k = args.k
    game_rounds = k * 200
    if args.rounds is not None:
        game_rounds = args.rounds

    print(
        f"Running experiment for network \"{args.network}\" with game_rounds={game_rounds} k={k} and memory={memory}\n.\n.\n.")

    # Run the game
    game = Game(s, A, L, k)
    result = game.run(game_rounds, memory)

    # Save the result
    experiment = args.experiment
    exportGameResult(args.network, game, result, k, memory, experiment)


if __name__ == "__main__":
    main()
