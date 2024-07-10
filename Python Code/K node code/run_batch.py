import argparse
import os
from utils import *


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("network",
                        help="The network to run the experiment on")
    parser.add_argument("k",
                        help="The number of nodes to select",
                        type=int)
    parser.add_argument("-r", "--rounds",
                        help="The number of rounds to run the game for. Default is k * 200",
                        type=int)
    parser.add_argument("-m", "--memory",
                        help="The memory of the game. Default is 10",
                        type=int,
                        default=10)
    args = parser.parse_args()

    # Configure the game
    memory = args.memory
    k = args.k
    game_rounds = k * 200
    if args.rounds is not None:
        game_rounds = args.rounds

    print(
        f"Running experiment for network \"{args.network}\" with game_rounds={game_rounds} k={k} and memory={memory}...")

    for i in range(1, k + 1):
        print(f'SELECTING {i} NODES')
        print('_' * 20)

        for experiment in range(1, 11):
            print(f'Experiment {experiment} k={k}')
            print('_' * 20)

            # Run the game
            os.system(
                f"python run.py {args.network} {k} {experiment} -r {game_rounds} -m {memory}")


if __name__ == "__main__":
    main()
