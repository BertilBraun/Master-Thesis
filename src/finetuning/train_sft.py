# The script is then responsible to train the current model 'current-finetuned-model' on the Dataset
# - apparently 3+ epochs are possible (https://x.com/rm_rafailov/status/1751738917613912086)

import gc
import os
import sys
import logging
import multiprocessing

import torch
import datasets
import transformers
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


from src.finetuning.train import load_dataset
from src.finetuning.defines import *

from src.finetuning.util.log_gpu_usage import trace_gpu_usage


os.environ['WANDB_DISABLED'] = 'true'

TEST_PERCENTAGE = 0.002

TRAINING_OUTPUT_DIR = f'{OUTPUT_DIR}/training'

NUMBER_OF_EPOCHS_TO_TRAIN = 4


if __name__ == '__main__':
    START_DATETIME = get_previous_datetime_str()


logger = logging.getLogger(__name__)


###################
# Hyper-parameters
###################
train_conf = SFTConfig(
    bf16=True,
    do_eval=False,
    learning_rate=1.0e-05,
    log_level='info',
    logging_steps=1,
    logging_strategy='steps',
    lr_scheduler_type='cosine',
    num_train_epochs=2,
    max_steps=-1,
    output_dir='./checkpoint_dir',
    overwrite_output_dir=True,
    per_device_eval_batch_size=2,
    per_device_train_batch_size=2,
    remove_unused_columns=True,
    save_steps=100,
    save_total_limit=1,
    seed=0,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    gradient_accumulation_steps=8,
    warmup_ratio=0.1,
    report_to=['tensorboard'],  # report metrics to tensorboard
)
peft_conf = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules='all-linear',
    modules_to_save=None,
)


###############
# Setup logging
###############
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f'Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}'
    + f' distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}'
)
logger.info(f'Training/evaluation parameters {train_conf}')
logger.info(f'PEFT parameters {peft_conf}')


################
# Model Loading
################
checkpoint_path = 'microsoft/Phi-3-mini-128k-instruct'
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
    device_map='auto',
    use_cache=False,
    attn_implementation='flash_attention_2',
    torch_dtype='auto',
    load_in_8bit=True,
    original_max_position_embeddings=8192 * 2,
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 8096 * 2
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'


##################
# Data Processing
##################
def apply_chat_template(example):
    example['text'] = example['prompt'] + example['chosen']
    return example


print('Loading datasets')
train_dataset, test_dataset = load_dataset(get_preference_output_file_path(START_DATETIME))


def len_of_input(text) -> int:
    return len(tokenizer(text)['input_ids'])  # type: ignore


# lets find the max length of the prompt
print('Finding dataset lengths')
prompt_lengths = [len_of_input(x) for x in train_dataset['prompt']]
chosen_lengths = [len_of_input(x) for x in train_dataset['chosen']]
max_seq_length = max(prompt + chosen for prompt, chosen in zip(prompt_lengths, chosen_lengths))
print(f'max prompt + chosen length: {max_seq_length}')


column_names = list(train_dataset.features)

processed_train_dataset = train_dataset.map(
    apply_chat_template,
    num_proc=10,
    remove_columns=column_names,
    desc='Applying chat template to train_sft',
)

processed_test_dataset = test_dataset.map(
    apply_chat_template,
    num_proc=10,
    remove_columns=column_names,
    desc='Applying chat template to test_sft',
)


###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    max_seq_length=max_seq_length,
    dataset_text_field='text',
    tokenizer=tokenizer,
)

# start log_gpu_usage() here on a separate thread
thread = multiprocessing.Process(
    target=trace_gpu_usage, args=(f'{OUTPUT_DIR}/GPU_usage_during_sft_training_{START_DATETIME}.txt',)
)
thread.start()

try:
    train_result = trainer.train()  # type: ignore
except Exception:
    thread.terminate()
    exit()
finally:
    thread.terminate()


# ############
# # Save model
# ############
trainer.save_model(train_conf.output_dir)

# free the memory again
del model
del trainer

gc.collect()
torch.cuda.empty_cache()

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    train_conf.output_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(train_conf.output_dir + '/merged', safe_serialization=True, max_shard_size='2GB')
