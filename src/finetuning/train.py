# The script is then responsible to train the current model 'current-finetuned-model' on the Dataset
# - apparently 3+ epochs are possible (https://x.com/rm_rafailov/status/1751738917613912086)


import gc
import multiprocessing
from torch import cuda, float16

from datasets import Dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast
from peft import AutoPeftModelForCausalLM, LoraConfig
from trl import DPOTrainer, DPOConfig
from src.finetuning.logic.finetuning_types import PreferenceSample
from src.finetuning.logic.tokenizer import get_tokenizer
from src.finetuning.util.log_gpu_usage import trace_gpu_usage
from src.finetuning.defines import *
from src.util.files import create_folder_backup
from src.util.json import load_json

TEST_PERCENTAGE = 0.002

TRAINING_OUTPUT_DIR = f'{OUTPUT_DIR}/training'

NUMBER_OF_EPOCHS_TO_TRAIN = 6


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

    for sample in load_json(json_dataset_file, PreferenceSample):
        prompts.append(sample.prompt)
        chosens.append(sample.chosen.removeprefix('Domain: "'))
        rejecteds.append(sample.rejected.removeprefix('Domain: "'))

    ds = Dataset.from_dict({'prompt': prompts, 'chosen': chosens, 'rejected': rejecteds})

    ds = ds.train_test_split(test_size=test_percentage, shuffle=True)

    return ds['train'], ds['test']


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
        learning_rate=5e-6,  # 4x higher LR than QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.1,  # warmup ratio based on QLoRA paper
        lr_scheduler_type='cosine',  # use cosine learning rate scheduler
        logging_steps=1,  # log every step
        save_steps=60,  # when to save checkpoint # approx every 30min
        save_total_limit=2,  # limit the total amount of checkpoints
        evaluation_strategy='steps',  # evaluate every 1000 steps
        eval_steps=2400,  # when to evaluate # approx every 2hours
        bf16=True,  # use bfloat16 precision
        fp16=False,  # use fp16 precision
        tf32=False,  # use tf32 precision
        # Currently crashes training group_by_length=True,  # group samples by length for faster training
        push_to_hub=False,  # push model to hub
        report_to=['tensorboard'],  # report metrics to tensorboard
        log_level='info',
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
        CURRENT_MODEL_PATH + '_run_3',
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


if __name__ == '__main__':
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
