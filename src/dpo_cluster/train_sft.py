import sys
import logging

import datasets
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


# The script is then responsible to train the current model 'current-finetuned-model' on the Dataset
# - apparently 3+ epochs are possible (https://x.com/rm_rafailov/status/1751738917613912086)


from dataclasses import dataclass
import os
import json
import shutil
from typing import Any

from datasets import Dataset

os.environ['WANDB_DISABLED'] = 'true'


TEST_PERCENTAGE = 0.00
# WARNING there is a copy of this variable in src/dpo_cluster/train.py
OUTPUT_DIR = '../../dpo_output'


TRAINING_OUTPUT_DIR = f'{OUTPUT_DIR}/training'
# WARNING there is a copy of this variable in src/dpo_cluster/train.py
CURRENT_MODEL_PATH = f'./{OUTPUT_DIR}/current-finetuned-model'

# WARNING there is a copy of this variable in src/dpo_cluster/train.py
BASE_MODEL_ID = 'microsoft/Phi-3-mini-128k-instruct'

NUMBER_OF_EPOCHS_TO_TRAIN = 4


def get_preference_output_file_path(start_datetime: str) -> str:
    # WARNING there is a copy of this function in src/dpo_cluster/train.py
    return f'{OUTPUT_DIR}/preferences/{start_datetime}.json'


def get_previous_datetime_str() -> str:
    # WARNING there is a copy of this function in src/dpo_cluster/train.py
    assert os.path.exists(
        f'{OUTPUT_DIR}/start_datetime.txt'
    ), 'Run get_new_datetime_str() first to create the start_datetime.txt file.'
    with open(f'{OUTPUT_DIR}/start_datetime.txt', 'r') as f:
        return f.read()


@dataclass(frozen=True)
class PreferenceSample:
    # WARNING there is a copy of this class in src/dpo_cluster/train.py
    prompt: str
    chosen: str
    rejected: str

    @staticmethod
    def from_json(data: dict) -> 'PreferenceSample':
        return PreferenceSample(**data)


def load_json(file_name: str) -> Any:
    if not os.path.exists(file_name):
        print(f'File does not exist: {file_name}')
        exit(1)
    with open(file_name, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    START_DATETIME = get_previous_datetime_str()


def load_dataset(json_dataset_file: str, test_percentage: float = TEST_PERCENTAGE) -> tuple[Dataset, Dataset]:
    """Load the dataset from the database and split it into training and test sets.
    The dataset is preprocessed using the tokenizer and contains the following columns:
    - prompt: list[str]
    - chosen: list[str]
    - rejected: list[str]

    Returns:
        (Dataset, Dataset): The training and test datasets.
    """

    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []

    for element in load_json(json_dataset_file):
        sample = PreferenceSample.from_json(element)
        prompts.append(sample.prompt)
        chosens.append(sample.chosen.removeprefix('Domain: "'))
        rejecteds.append(sample.rejected.removeprefix('Domain: "'))

    ds = Dataset.from_dict({'prompt': prompts, 'chosen': chosens, 'rejected': rejecteds})

    ds = ds.train_test_split(test_size=test_percentage, shuffle=True)

    return ds['train'], ds['test']


def create_folder_backup(folder_path: str) -> tuple[bool, str]:
    if not os.path.exists(folder_path):
        print(f'No folder to backup found at {folder_path}')
        return False, ''

    backup_path = folder_path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copytree(folder_path, backup_path)
        return True, backup_path

    for i in range(1, 1000):
        backup_path = folder_path + f'.bak({i})'
        if not os.path.exists(backup_path):
            shutil.copytree(folder_path, backup_path)
            return True, backup_path

    print(f'Could not create backup for {folder_path}')
    return False, ''


logger = logging.getLogger(__name__)


###################
# Hyper-parameters
###################
train_conf = SFTConfig(
    bf16=True,
    do_eval=False,
    learning_rate=5.0e-06,
    log_level='info',
    logging_steps=20,
    logging_strategy='steps',
    lr_scheduler_type='cosine',
    num_train_epochs=2,
    max_steps=-1,
    output_dir='./checkpoint_dir',
    overwrite_output_dir=True,
    per_device_eval_batch_size=4,
    per_device_train_batch_size=4,
    remove_unused_columns=True,
    save_steps=100,
    save_total_limit=1,
    seed=0,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    gradient_accumulation_steps=1,
    warmup_ratio=0.2,
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
    use_cache=False,
    trust_remote_code=True,
    attn_implementation='flash_attention_2',  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map=None,
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
    packing=True,
)
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics('train', metrics)
trainer.save_metrics('train', metrics)
trainer.save_state()


#############
# Evaluation
#############
# tokenizer.padding_side = 'left'
# metrics = trainer.evaluate()
# metrics['eval_samples'] = len(processed_test_dataset)
# trainer.log_metrics('eval', metrics)
# trainer.save_metrics('eval', metrics)


# ############
# # Save model
# ############
trainer.save_model(train_conf.output_dir)
