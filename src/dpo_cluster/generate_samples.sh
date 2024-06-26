#!/bin/bash

#SBATCH --job-name=generate                # job name
#SBATCH --partition=gpu_8                  # mby GPU queue for the resource allocation.
#SBATCH --time=01:00:00                    # wall-clock time limit
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=8                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:2 # TODO 8
#SBATCH --output=generate_%j.txt
#SBATCH --error=generate_%j.txt

source shared_slurm_setup.sh

python -m src.dpo_cluster.generate_samples

# if the generate script is successful, then the next step is to train the model
if [ $? -eq 0 ]; then
    cd src/dpo_cluster
    sbatch evaluate_samples.sh
fi