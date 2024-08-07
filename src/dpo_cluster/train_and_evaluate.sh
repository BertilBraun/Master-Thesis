#!/bin/bash

#SBATCH --job-name=train                   # job name
#SBATCH --partition=gpu_4_a100             # mby GPU queue for the resource allocation.
#SBATCH --time=08:00:00                    # wall-clock time limit 
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=4                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=train_%j.txt
#SBATCH --error=train_%j.txt

source shared_slurm_setup.sh

cd src/dpo_cluster
# accelerate launch --num_processes=2 train.py
python train.py


if [ $? -eq 0 ]; then
    cd ../..

    python -m src.dpo_cluster.evaluate_model_after_finetuning mlpc

    # if the train script is successful, then the next step is to generate again
    if [ $? -eq 0 ]; then
        cd src/dpo_cluster
        # TODO reactivate sbatch generate.sh
    fi
fi
