#!/bin/bash

#SBATCH --job-name=evaluate_llama3-70B         # job name
#SBATCH --partition=single                 # mby GPU queue for the resource allocation.
#SBATCH --time=02:00:00                    # wall-clock time limit
#SBATCH --mem=5000                         # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=1                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --output=evaluate_llama3-70B_%j.txt
#SBATCH --error=evaluate_llama3-70B_%j.txt

source shared_slurm_setup.sh

python -m src.dpo_cluster.evaluate_samples_via_api dev-llama-large mlpc none

# if the generate script is successful, then the next step is to train the model
if [ $? -eq 0 ]; then
    cd src/dpo_cluster
    sbatch train_and_evaluate.sh
fi