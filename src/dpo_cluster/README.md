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

To run the code, start with the `setup.sh` script. This will setup the model and evaluation etc. After this, you can run the `generate_samples.sh` script to extract the profiles, then run `evaluate_samples.sh` to generate the preference data. The `train_and_evaluate.sh` script will train and evaluate the model.

```bash
cd ~/Master-Thesis/src/dpo_cluster
# Run first setup, then generate and finally train
./setup.sh
./generate_samples.sh
./evaluate_samples.sh
./train_and_evaluate.sh
```

## Queuing jobs

To properly run the fine-tuning on the cluster, you will need to queue the jobs with multiple GPUs. To do so, run the following commands:

```bash
cd ~/Master-Thesis/src/dpo_cluster
sbatch setup.sh
```

## Scaling calculations and notes

how many samples to we generate with each extraction of TOP_K_TO_SAMPLE?

- 4 papers per sample
- TOP_K_TO_SAMPLE extracted profiles
- TOP_K_TO_SAMPLE profiles in a tournament
- TOP_K_TO_SAMPLE - 1 comparisons in a tournament
- TOP_K_TO_SAMPLE = 8 -> 12 usable preferences and 7 comparisons
- TOP_K_TO_SAMPLE = 16 -> 32 usable preferences and 15 comparisons

=> higher TOP_K_TO_SAMPLE means more usable preferences with comparativly less comparisons
   but limited by the number of good profiles we can extract with such a high TEMPERATURE

TODO are the TOP_K_TO_SAMPLE samples different enough?

TODO how long does extracting NUM_SAMPLES_TO_GENERATE samples take? Measure it!
Theoretically:

- NUM_SAMPLES_TO_GENERATE samples / 32 preferences = 63 tournaments
- 63 tournaments \* 15 comparisons = 945 comparisons
- 945 comparisons \* 30 seconds / NUM_THREADS_EVALUATE = 1.6 hours
- 63 extractions \* 30 seconds \* TOP_K_TO_SAMPLE / NUM_THREADS_GENERATE = 2.8 hours

TODO how do generating and evaluating compare in time? Do we need more threads for one or the other?

How much would 10k training samples cost?

- Approximately 3.0k Tokens in a one-shot prompt
- ~300 tokens for the response
- 1M tokens input = 5\$
- 1M tokens output = 15\$
- 945 comparisons \* 3.0k tokens = 2.8M tokens => 2.8M tokens \* 5\$/1M tokens = 14\$ for input
- 945 comparisons \* 300 tokens = 283.5k tokens => 283.5k tokens \* 15\$/1M tokens = 4.25\$ for output
- ~19$ per 2000 Samples
- Can be cut to 14\$ with 1x batching
  - 19$ * 3/4 = ~14\$
  - since half of the comparisons are in the first round and these would be batched with half the price
  - 1 day waiting
- Can be cut to 12\$ with 2x batching
  - 19$ * 5/8 = ~12\$
  - batching the first round and then also the second round
  - 2 days waiting

## Lots of references

**Nodes on the BWUniCluster:**

<https://wiki.bwhpc.de/e/BwUniCluster2.0/Hardware_and_Architecture>
<https://wiki.bwhpc.de/e/BwUniCluster2.0/Batch_Queues>

**LLM generation:**

<https://huggingface.co/docs/transformers/en/llm_tutorial>

**LLM.generate() parameters:**

<https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.GenerationConfig>

**Memory Usage:**

<https://huggingface.co/spaces/hf-accelerate/model-memory-usage>

Example model: <https://huggingface.co/instruction-pretrain/finance-Llama3-8B>

Training is for Adam, though idk. we would use LoRA, which should reduce the gradient and backward pass memory usage by a lot

**LLM fine-tuning:**

- <https://huggingface.co/docs/transformers/perf_train_gpu_one>

- Faster training using `accelerate` and `deepspeed`:
  - <https://huggingface.co/docs/trl/v0.9.4/customization>
  - <https://huggingface.co/docs/accelerate/usage_guides/deepspeed>

- <https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/README.md>
- <https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/sft_llama2.py>
- <https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py>

- <https://huggingface.co/blog/rlhf>
- <https://huggingface.co/blog/dpo-trl>
- <https://huggingface.co/blog/pref-tuning>
- <https://huggingface.co/docs/trl/en/dpo_trainer>
- <https://huggingface.co/docs/peft/quicktour>
- <https://huggingface.co/docs/peft/main/en/conceptual_guides/lora>

- <https://gitlab.kit.edu/kit/aifb/BIS/templates/projects/llmfinetuningstarterkit/-/blob/main/supervised_finetuning/finetuning.ipynb?ref_type=heads>

- <https://github.com/huggingface/alignment-handbook> - Parameters for multiple different model trainings
