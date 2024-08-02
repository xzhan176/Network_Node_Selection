import argparse
import os
import scipy
from os import cpu_count
from game import Game, exportGameResult
from utils import *


def run(network: str, k: int, experiment: int, memory: int,
        game_rounds: int | None = None,
        disk_dumped: bool = False,
        zero_sum: bool = False):
    # Prepare network
    if not disk_dumped:
        network_module = import_network(network)
        G, s, n = network_module.init()
        L = scipy.sparse.csgraph.laplacian(G, normed=False)
        A = np.linalg.inv(np.identity(n) + L)
    else:
        s = None
        A = None

    if game_rounds is None:
        game_rounds = k * 200

    print('-' * 40)
    print(
        f"Running experiment {experiment} for network \"{network}\" (n = {n}) with game_rounds={game_rounds} k={k} memory={memory}.")
    print(f'Number of cores available: {cpu_count()}.')
    print('\n.\n.\n.')

    # Run the game
    game = Game(k, s, A,
                disk_dumped=disk_dumped,
                zero_sum=zero_sum,
                temp_path=os.path.join(
                    os.path.dirname(__file__),
                    "memmaps",
                    f"memmaps_{network}_{k}_{experiment}"))
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
    parser.add_argument("-dd", "--disk-dumped",
                        help="Use disk dumped data",
                        action="store_true")
    parser.add_argument("-z", "--zero-sum",
                        help="Whether the game is zero-sum",
                        action="store_true",
                        default=False)
    args = parser.parse_args()

    fn_benchmark(lambda: run(
        args.network,
        args.k,
        args.experiment,
        args.memory,
        game_rounds=args.rounds,
        disk_dumped=args.disk_dumped,
        zero_sum=args.zero_sum))


if __name__ == "__main__":
    main()
