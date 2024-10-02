#!/bin/bash

#SBATCH --job-name=train_sft               # job name
#SBATCH --partition=gpu_4_a100             # mby GPU queue for the resource allocation.
#SBATCH --time=11:00:00                    # wall-clock time limit 
#SBATCH --mem=100000                       # memory per node
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --cpus-per-task=4                  # number of CPUs required per MPI task
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.
#SBATCH --gres=gpu:1
#SBATCH --output=train_sft_%j.txt
#SBATCH --error=train_sft_%j.txt

source shared_slurm_setup.sh

cd src/finetuning
python train_sft.py
