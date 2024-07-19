#!/bin/bash

#SBATCH --job-name=evaluate_gpt-4o-mini         # job name
#SBATCH --partition=multiple               # mby GPU queue for the resource allocation.
#SBATCH --time=02:00:00                    # wall-clock time limit
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=20                 # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --output=evaluate_gpt-4o-mini_%j.txt
#SBATCH --error=evaluate_gpt-4o-mini_%j.txt

source shared_slurm_setup.sh

python -m src.dpo_cluster.evaluate_samples_via_api gpt-4o-mini openai CAS

# if the generate script is successful, then the next step is to train the model
if [ $? -eq 0 ]; then
    cd src/dpo_cluster
    sbatch train_and_evaluate.sh
fi