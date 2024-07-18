import argparse
import scipy
from game import Game, exportGameResult
from utils import *


def run(network: str, k: int, experiment: int, memory: int, game_rounds: int | None = None):
    # Prepare network
    network_module = import_network(network)
    G, s, n = network_module.init()
    L = scipy.sparse.csgraph.laplacian(G, normed=False)
    A = np.linalg.inv(np.identity(n) + L)

    if game_rounds is None:
        game_rounds = k * 200

    print('-' * 40)
    print(
        f"Running experiment {experiment} for network \"{network}\" (n = {n}) with game_rounds={game_rounds} k={k} memory={memory}\n.\n.\n.")

    # Run the game
    game = Game(s, A, L, k)
    result = game.run(game_rounds, memory)

    # Save the result
    exportGameResult(network, game, result, k, memory, experiment)


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

    fn_benchmark(lambda: run(
        args.network, args.k, args.experiment, args.memory, args.rounds))


if __name__ == "__main__":
    main()
