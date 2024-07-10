import argparse
import os
from utils import *


def generateScriptContent(network, k, experiment, game_rounds, memory):
    job_name = f"opinion-polarization-{network}-k-{k}-experiment-{experiment}-memory-{memory}"
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=results/{job_name}-output.txt
#SBATCH --error=results/{job_name}-error.log
#SBATCH --ntasks=1

python run.py {network} {k} {experiment} -r {game_rounds} -m {memory}
"""
    return script


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
    parser.add_argument("--no-slurm",
                        help="Don't use SLURM to run the experiments",
                        action="store_true")
    args = parser.parse_args()

    # Configure the game
    memory = args.memory
    k = args.k
    game_rounds = k * 200
    if args.rounds is not None:
        game_rounds = args.rounds

    print(
        f"Running experiment for network \"{args.network}\" with game_rounds={game_rounds} k={k} and memory={memory}...")

    temp_script = "run_batch.sh"

    # Run the game
    for i in range(1, k + 1):
        print(f'SELECTING {i} NODES')
        print('_' * 20)

        for experiment in range(1, 11):
            print(f'Experiment {experiment} k={k}')
            print('_' * 20)

            if args.no_slurm:
                os.system(
                    f"python run.py {args.network} {k} {experiment} -r {game_rounds} -m {memory}")
            else:
                f = open(temp_script, "w")
                f.write(generateScriptContent(args.network,
                        k, experiment, game_rounds, memory))
                f.close()
                os.system(f"sbatch {temp_script}")
    try:
        os.remove(temp_script)
    except FileNotFoundError:
        # file wasn't created if the script wasn't run in SLURM mode
        pass


if __name__ == "__main__":
    main()
