# The script is then responsible to train the current model 'current-finetuned-model' on the Dataset
# - apparently 3+ epochs are possible (https://x.com/rm_rafailov/status/1751738917613912086)


from dataclasses import dataclass
import gc
import os
import json
import GPUtil
import shutil
import multiprocessing
from time import sleep
from typing import Any
from torch import cuda, float16

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast
from peft import AutoPeftModelForCausalLM, LoraConfig
from trl import DPOTrainer, DPOConfig


TEST_PERCENTAGE = 0.05
# WARNING there is a copy of this variable in src/dpo_cluster/train.py
OUTPUT_DIR = '../../dpo_output'


TRAINING_OUTPUT_DIR = f'{OUTPUT_DIR}/training'
# WARNING there is a copy of this variable in src/dpo_cluster/train.py
CURRENT_MODEL_PATH = f'./{OUTPUT_DIR}/current-finetuned-model'

# WARNING there is a copy of this variable in src/dpo_cluster/train.py
BASE_MODEL_ID = 'microsoft/Phi-3-mini-128k-instruct'

NUMBER_OF_EPOCHS_TO_TRAIN = 4


def trace_gpu_usage(file_name: str):
    average_usage = 0
    average_memory_free = 0
    average_memory_used = 0
    average_memory_total = 0
    average_temperature = 0
    total_num_samples = 0

    with open(file_name, 'w') as f:
        while True:
            for gpu in GPUtil.getGPUs():
                f.write(f'GPU ID: {gpu.id}, Name: {gpu.name}\n')
                f.write(f'Belastung: {gpu.load * 100:.1f}%\n')
                f.write(f'Freier Speicher: {gpu.memoryFree:.2f} MB\n')
                f.write(f'Verwendeter Speicher: {gpu.memoryUsed:.2f} MB\n')
                f.write(f'Gesamtspeicher: {gpu.memoryTotal:.2f} MB\n')
                f.write(f'Temperatur: {gpu.temperature:.2f} C\n')
                f.write('-' * 40 + '\n')
                average_usage += gpu.load * 100
                average_memory_free += gpu.memoryFree
                average_memory_used += gpu.memoryUsed
                average_memory_total += gpu.memoryTotal
                average_temperature += gpu.temperature
                total_num_samples += 1
            f.write(f'Average usage: {average_usage / total_num_samples:.1f}%\n')
            f.write(f'Average memory free: {average_memory_free / total_num_samples:.2f} MB\n')
            f.write(f'Average memory used: {average_memory_used / total_num_samples:.2f} MB\n')
            f.write(f'Average memory total: {average_memory_total / total_num_samples:.2f} MB\n')
            f.write(f'Average temperature: {average_temperature / total_num_samples:.2f} C\n')
            f.write('=' * 40 + '\n')
            f.flush()

            sleep(2)


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


def get_tokenizer(name_or_path: str = BASE_MODEL_ID) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    # WARNING there is a copy of this function in src/dpo_cluster/train.py
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        padding_side='left',
        truncation_side='left',
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<|' + message['role'] + '|> ' + message['content'] + '<|eos|>\n'}}{% endfor %}"
    tokenizer.model_max_length = 8096 * 2

    return tokenizer


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


def get_trainer(
    model,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_seq_length: int,
    prompt_length: int,
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> DPOTrainer:
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules='all-linear',
    )

    args = DPOConfig(
        output_dir=TRAINING_OUTPUT_DIR,  # directory to save and repository id
        num_train_epochs=NUMBER_OF_EPOCHS_TO_TRAIN,  # number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=2,  # batch size for evaluation
        gradient_accumulation_steps=8,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim='adamw_torch_fused',  # use fused adamw optimizer
        learning_rate=1e-5,  # 4x higher LR than QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.1,  # warmup ratio based on QLoRA paper
        lr_scheduler_type='cosine',  # use cosine learning rate scheduler
        logging_steps=1,  # log every step
        save_steps=60,  # when to save checkpoint # approx every 30min
        save_total_limit=2,  # limit the total amount of checkpoints
        evaluation_strategy='steps',  # evaluate every 1000 steps
        eval_steps=240,  # when to evaluate # approx every 2hours
        bf16=True,  # use bfloat16 precision
        fp16=False,  # use fp16 precision
        tf32=False,  # use tf32 precision
        # Currently crashes training group_by_length=True,  # group samples by length for faster training
        push_to_hub=False,  # push model to hub
        report_to=['tensorboard'],  # report metrics to tensorboard
        max_length=max_seq_length,
        max_prompt_length=prompt_length,
        remove_unused_columns=False,
    )

    return DPOTrainer(
        model,
        ref_model=None,  # set to none since we use peft
        peft_config=peft_config,  # type: ignore
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        beta=0.1,  # The beta factor in DPO loss. Higher beta means less divergence
        loss_type='sigmoid',  # The loss type for DPO.
    )


def get_model_to_train():
    # Load model and tokenizer
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,  # TODO to continue training the current model, set back to CURRENT_MODEL_PATH,
        device_map='auto',
        use_cache=False,
        attn_implementation='flash_attention_2',
        torch_dtype='auto',
        load_in_8bit=True,
        original_max_position_embeddings=8192 * 2,
    )


def merge_and_save_model():
    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        TRAINING_OUTPUT_DIR,
        torch_dtype=float16,
        low_cpu_mem_usage=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(CURRENT_MODEL_PATH, safe_serialization=True, max_shard_size='2GB')


def main():
    # Load all the database datasets into one Dataset
    print('Loading datasets')
    train_dataset, test_dataset = load_dataset(get_preference_output_file_path(START_DATETIME))

    tokenizer = get_tokenizer()

    def len_of_input(text) -> int:
        return len(tokenizer(text)['input_ids'])  # type: ignore

    # lets find the max length of the prompt
    print('Finding dataset lengths')
    prompt_lengths = [len_of_input(x) for x in train_dataset['prompt']]
    chosen_lengths = [len_of_input(x) for x in train_dataset['chosen']]
    rejected_lengths = [len_of_input(x) for x in train_dataset['rejected']]
    prompt_length = max(prompt_lengths)
    max_seq_length = max(
        max(prompt + chosen, prompt + rejected)
        for prompt, chosen, rejected in zip(prompt_lengths, chosen_lengths, rejected_lengths)
    )
    print(f'max prompt length: {prompt_length}')
    print(f'max prompt + chosen length: {max_seq_length}')

    print('Loading model')
    model = get_model_to_train()

    print('Loading trainer')
    trainer = get_trainer(model, tokenizer, max_seq_length, prompt_length, train_dataset, test_dataset)

    # start log_gpu_usage() here on a separate thread
    thread = multiprocessing.Process(
        target=trace_gpu_usage, args=(f'{OUTPUT_DIR}/GPU_usage_during_training_{START_DATETIME}.txt',)
    )
    thread.start()

    create_folder_backup(CURRENT_MODEL_PATH)

    try:
        with cuda.amp.autocast():
            trainer.train()

        # save model at the end of training
        trainer.save_model()
    finally:
        thread.terminate()

    # free the memory again
    del model
    del trainer

    gc.collect()
    cuda.empty_cache()

    merge_and_save_model()


if __name__ == '__main__':
    main()
