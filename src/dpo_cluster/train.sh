#!/bin/bash

#SBATCH --job-name=train                   # job name
#SBATCH --partition=gpu_4                  # mby GPU queue for the resource allocation.
#SBATCH --time=04:00:00                    # wall-clock time limit
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=train_%j.txt
#SBATCH --error=train_%j.txt

source shared_slurm_setup.sh

python -m src.dpo_cluster.train

# if the train script is successful, then the next step is to generate again
if [ $? -eq 0 ]; then
    cd src/dpo_cluster
    sbatch generate.sh
fi