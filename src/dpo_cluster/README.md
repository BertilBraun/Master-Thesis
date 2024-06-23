# Instructions for the BwUniCluster

## Setup

Connect to the cluster and ensure, that you have a GPU available. You can check this by running `nvidia-smi`. Tipp: Use the jupyter notebook interface to run the code, there you can select the GPU in the settings and create as many terminals as you like.

After this, you can clone the repository and install the requirements:

```bash
git clone https://github.com/BertilBraun/Master-Thesis.git ~/Master-Thesis
cd ~/Master-Thesis/src/dpo_cluster
source cluster_setup.sh
```

## Run the code

To run the code, start with the `setup.sh` script. This will setup the model and evaluation etc. After this, you can run the `generate.sh` script to generate the data. The `train.sh` script will train and evaluate the model.

```bash
cd ~/Master-Thesis/src/dpo_cluster
# Run first setup, then generate and finally train
./setup.sh
./generate.sh
./train.sh
```

## Queuing jobs

To properly run the fine-tuning on the cluster, you will need to queue the jobs with multiple GPUs. To do so, run the following commands:

```bash
cd ~/Master-Thesis/src/dpo_cluster
sbatch setup.sh
```
