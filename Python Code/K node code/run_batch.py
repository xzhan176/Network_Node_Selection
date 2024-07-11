import argparse
import os
from utils import *


def generateScriptContent(network, k, experiment, game_rounds, memory):
    job_name = f"{network[:2]}k{k}e{experiment}"
    result_name = f"network-{network}-k-{k}-experiment-{experiment}-memory-{memory}"
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=results/{result_name}-output.txt
#SBATCH --error=results/{result_name}-error.log
#SBATCH --ntasks=1

python run.py {network} {k} {experiment} -r {game_rounds} -m {memory}
"""
    return script


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Submit a batch of experiments with number of nodes range from 1 to k")
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
    maxK = args.k
    if args.rounds is not None:
        game_rounds = args.rounds

    temp_script = "run_batch.sh"

    # Run the games
    for k in range(1, maxK + 1):
        game_rounds = k * 200

        for experiment in range(1, 11):
            print('-' * 20, flush=True)
            print(f'Experiment {experiment} k={k}', flush=True)

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
