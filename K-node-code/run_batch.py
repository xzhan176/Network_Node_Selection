import argparse
import os
import scipy
from game import Game
from utils import *


def generateScriptContent(network, k, experiment, game_rounds, memory, zero_sum, cpu_count=1):
    job_name = f"{network[:2]}k{k}e{experiment}"
    result_name = f"{network}-k-{k}-e-{experiment}-m-{memory}"
    zero_sum_argument = "--zero-sum" if zero_sum else ""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=results/{result_name}-output.txt
#SBATCH --error=results/{result_name}-error.log
#SBATCH --time=122:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpu_count}

python run.py {network} {k} {experiment} -r {game_rounds} -m {memory} {zero_sum_argument}
"""
    return script


def main():
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Submit a batch of experiments with number of nodes range from 1 to k")
    parser.add_argument("network",
                        help="The network to run the experiment on")
    parser.add_argument("k",
                        help="The number of nodes to select. The script will run experiments for each k=1,2,3,...,k.",
                        type=int)
    parser.add_argument("-r", "--rounds",
                        help="The number of rounds to run the game for. Default is k * 200",
                        type=int)
    parser.add_argument("-e", "--experiments",
                        help="The number of experiments to run for each k. Default is 10",
                        type=int,
                        default=10)
    parser.add_argument("-m", "--memory",
                        help="The memory of the game. Default is 10",
                        type=int,
                        default=10)
    parser.add_argument("-c", "--cpus-per-task",
                        help="The number of CPUs to use for each task. Default is 1",
                        type=int,
                        default=1)
    parser.add_argument("--no-slurm",
                        help="Don't use SLURM to run the experiments. Use this argument to run the script on a local machine.",
                        action="store_true",
                        default=False)
    parser.add_argument("-z", "--zero-sum",
                        help="Whether the game is zero-sum",
                        action="store_true",
                        default=False)
    args = parser.parse_args()

    # Configure the game
    memory = args.memory
    maxK = args.k

    # Preload network data to disk
    network_module = import_network(args.network)
    G, s, n = network_module.init()
    L = scipy.sparse.csgraph.laplacian(G, normed=False)
    A = np.linalg.inv(np.identity(n) + L)
    Game.loadDataToDisk(s, f"s_memmaps_{args.network}")
    Game.loadDataToDisk(A, f"A_memmaps_{args.network}")

    temp_script = "run_batch.sh"

    # Run the games
    for k in range(1, maxK + 1):
        game_rounds = k * 200 if args.rounds is None else args.rounds
        zero_sum_argument = "--zero-sum" if args.zero_sum else ""

        for experiment in range(1, args.experiments + 1):
            print('-' * 20, flush=True)
            print(f'Experiment {experiment} k={k}', flush=True)

            if args.no_slurm:
                python_script = f"python run.py {args.network} {k} {experiment} -r {game_rounds} -m {memory} {zero_sum_argument}"
                os.system(python_script)
            else:
                f = open(temp_script, "w")
                f.write(generateScriptContent(args.network,
                        k, experiment, game_rounds, memory,
                        zero_sum=args.zero_sum,
                        cpu_count=args.cpus_per_task))
                f.close()
                os.system(f"sbatch {temp_script}")
    try:
        os.remove(temp_script)
    except FileNotFoundError:
        # file wasn't created if the script wasn't run in SLURM mode
        pass


if __name__ == "__main__":
    main()
