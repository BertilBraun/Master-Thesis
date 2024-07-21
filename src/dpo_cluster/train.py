# The script is then responsible to train the current model 'current-finetuned-model' on the Dataset
# - apparently 3+ epochs are possible (https://x.com/rm_rafailov/status/1751738917613912086)


from dataclasses import dataclass
import gc
import json
import multiprocessing
import os
from typing import Any
from torch import bfloat16, float16, cuda

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig,
)
from peft import AutoPeftModelForCausalLM, LoraConfig
from trl import DPOTrainer, DPOConfig

from rich.console import Console
from rich.status import Status


TEST_PERCENTAGE = 0.05
# WARNING there is a copy of this variable in src/dpo_cluster/train.py
OUTPUT_DIR = os.path.expanduser('~/Master-Thesis/dpo_output')


TRAINING_OUTPUT_DIR = f'{OUTPUT_DIR}/training'
# WARNING there is a copy of this variable in src/dpo_cluster/train.py
CURRENT_MODEL_PATH = f'./{OUTPUT_DIR}/current-finetuned-model'

# WARNING there is a copy of this variable in src/dpo_cluster/train.py
BASE_MODEL_ID = 'microsoft/Phi-3-mini-4k-instruct'

NUMBER_OF_EPOCHS_TO_TRAIN = 3


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
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<|' + message['role'] + '|> ' + message['content'] + '<|eos|>\n'}}{% endfor %}"

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


def progress_status(message: str) -> Status:
    # Can be use like this:
    # with progress_status('Some message'):
    #     something that happens for some time and the message gets displayed in the meantime with a loading indicator
    return Console().status('[bold green]' + message)


if __name__ == '__main__':
    START_DATETIME = get_previous_datetime_str()


def load_dataset(
    json_dataset_file: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    test_percentage: float = TEST_PERCENTAGE,
) -> tuple[Dataset, Dataset]:
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
        chosens.append(sample.chosen)
        rejecteds.append(sample.rejected)

    ds = Dataset.from_dict({'prompt': prompts, 'chosen': chosens, 'rejected': rejecteds})

    def process(row):
        return {
            'prompt': row['prompt'],  # the prompt already has the chat template applied
            'chosen': tokenizer.apply_chat_template(
                [{'role': 'assistant', 'content': row['chosen']}],
                tokenize=False,
            ),
            'rejected': tokenizer.apply_chat_template(
                [{'role': 'assistant', 'content': row['rejected']}],
                tokenize=False,
            ),
        }

    ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

    ds = ds.train_test_split(test_size=test_percentage, shuffle=True)

    return ds['train'], ds['test']


tokenizer = get_tokenizer()


def len_of_input(text) -> int:
    return len(tokenizer(text)['input_ids'])  # type: ignore


def find_p95_length(train_dataset: Dataset) -> tuple[int, int]:
    from numpy import percentile

    prompt_length = int(percentile([len_of_input(x) for x in train_dataset['prompt']], 95))
    max_seq_length_chosen = int(percentile([len_of_input(x['prompt'] + x['chosen']) for x in train_dataset], 95))  # type: ignore
    max_seq_length_rejected = int(percentile([len_of_input(x['prompt'] + x['rejected']) for x in train_dataset], 95))  # type: ignore
    max_seq_length = max(max_seq_length_chosen, max_seq_length_rejected)

    # Up the lengths to next multiple of 2, why 2? Don't know
    prompt_length = ((prompt_length + 1) // 2) * 2
    max_seq_length = ((max_seq_length + 1) // 2) * 2

    return prompt_length, max_seq_length


def filter_by_max_length(dataset: Dataset, max_length: int) -> Dataset:
    return dataset.filter(lambda x: len_of_input(x['prompt'] + x['chosen']) <= max_length)


def get_trainer(model) -> DPOTrainer:
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
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=4,  # batch size for evaluation
        gradient_accumulation_steps=4,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim='adamw_torch_fused',  # use fused adamw optimizer
        learning_rate=2e-5,  # 4x higher LR than QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.1,  # warmup ratio based on QLoRA paper
        lr_scheduler_type='cosine',  # use cosine learning rate scheduler
        logging_steps=25,  # log every 25 steps
        save_steps=500,  # when to save checkpoint
        save_total_limit=2,  # limit the total amount of checkpoints
        evaluation_strategy='steps',  # evaluate every 1000 steps
        eval_steps=700,  # when to evaluate
        bf16=True,  # use bfloat16 precision
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
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=bfloat16,
    )

    # Load model and tokenizer
    return AutoModelForCausalLM.from_pretrained(
        CURRENT_MODEL_PATH,
        device_map='auto',
        use_cache=False,
        attn_implementation='flash_attention_2',
        torch_dtype=bfloat16,
        quantization_config=bnb_config,
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
    merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size='2GB')


if __name__ == '__main__':
    # Load all the database datasets into one Dataset
    with progress_status('Loading datasets'):
        train_dataset, test_dataset = load_dataset(get_preference_output_file_path(START_DATETIME), tokenizer)

    # lets find the p95 length of the prompt
    with progress_status('Finding p95 lengths'):
        prompt_length, max_seq_length = find_p95_length(train_dataset)
    print(f'p95 prompt length: {prompt_length}')
    print(f'p95 prompt + chosen length: {max_seq_length}')

    # filter datasets to remove samples that are too long
    with progress_status('Filtering datasets'):
        old_len_train, old_len_test = len(train_dataset), len(test_dataset)
        train_dataset = filter_by_max_length(train_dataset, max_seq_length)
        test_dataset = filter_by_max_length(test_dataset, max_seq_length)
    print(f'len(train_dataset): {len(train_dataset)} -> removed {old_len_train - len(train_dataset)}')
    print(f'len(test_dataset): {len(test_dataset)} -> removed {old_len_test - len(test_dataset)}')

    with progress_status('Loading model'):
        model = get_model_to_train()

    with progress_status('Loading trainer'):
        trainer = get_trainer(model)

    trainer.train()

    # save model at the end of training
    trainer.save_model()

    # free the memory again
    del model
    del trainer

    gc.collect()
    cuda.empty_cache()

    merge_and_save_model()
