#!/bin/bash

#SBATCH --job-name=evaluate                # job name
#SBATCH --partition=dev_gpu_4_a100 # TODO gpu_4_a100             # mby GPU queue for the resource allocation.
#SBATCH --time=00:30:00  # TODO 3h?                  # wall-clock time limit
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=4                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:2 # TODO 4
#SBATCH --output=evaluate_%j.txt
#SBATCH --error=evaluate_%j.txt

source shared_slurm_setup.sh

python -m src.dpo_cluster.evaluate_samples

# if the generate script is successful, then the next step is to train the model
if [ $? -eq 0 ]; then
    cd src/dpo_cluster
    sbatch train_and_evaluate.sh
fi