# This script runs single-threaded for now

# We start by combining the threadlocal evaluation databases into a single Dataset which we can use in training

# The script is then responsible to train the current model 'current-finetuned-model' on the Dataset
# - apparently 3+ epochs are possible (https://x.com/rm_rafailov/status/1751738917613912086)

# We should then also evaluate on 10 samples to see if the model has improved
#   do this by having a list of (abstracts, best_profile_from_original_model, best_profile_from_last_model)
#   and then run the extraction on the abstracts with the current model
#   and compare the results to the best_profile_from_original_model and best_profile_from_last_model
#   and see if the current model has improved and write the results to a file
#   afterwards write the new best_profile_from_last_model to the file

# If the model is preferred less than the last model, we should stop the training and keep the last model
#   this is simply done by comparing the number of wins of the current model to the number of wins of the last model
#   if the current model has less wins, we stop the training and keep the last model
#   this is done by exiting the script with a non zero exit code


# Otherwise the script is then responsible to write the new model to the 'current-finetuned-model' file so that it can be used in the next iteration


import multiprocessing
import sys
import partialjson
from torch import bfloat16, float16, cuda

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BitsAndBytesConfig,
)
from peft import AutoPeftModelForCausalLM, LoraConfig
from trl import DPOTrainer, DPOConfig

from src.dpo_cluster.defines import *
from src.util import dump_json, load_json
from src.log import progress_status
from src.database import get_retriever_getter
from src.evaluation import default_round_evaluator, prompt_for_ranking, run_tournament_ranking
from src.types import EvaluationResult, Profile, Ranking


if __name__ == '__main__':
    START_DATETIME = get_previous_datetime_str()


def load_dataset(
    json_file_paths: list[str],
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

    for file in json_file_paths:
        for element in load_json(file):
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
        r=256,
        lora_alpha=128,
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


def get_model():
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


def evaluate_is_profile1_preferred(model, tokenizer, profile1: Profile, profile2: Profile, abstracts: list[str]) -> int:
    profiles = [profile1, profile2]

    examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking).invoke('\n\n'.join(abstracts))

    def evaluator(profile_index1: int, profile_index2: int) -> EvaluationResult:
        prompt_messages = prompt_for_ranking(profiles[profile_index1], profiles[profile_index2], examples, abstracts)
        prompt = prompt_messages_to_str(tokenizer, prompt_messages)

        response = generate(
            tokenizer,
            model,
            prompt,
            num_return_sequences=1,
            do_sample=False,
            max_new_tokens=350,
        )[0]

        return partialjson.JSONParser().parse(response)

    tournament = run_tournament_ranking(
        list(range(len(profiles))),
        default_round_evaluator(evaluator),
        do_shuffle=True,
    )

    return tournament.match.winner == 0


def evaluate_model() -> bool:
    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        TRAINING_OUTPUT_DIR,
        torch_dtype=float16,
        low_cpu_mem_usage=True,
    )

    samples_for_fine_tuning_improvement_evaluation: list[SampleForFineTuningImprovementEvaluation] = []

    for element in load_json(SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE):
        sample = SampleForFineTuningImprovementEvaluation.from_json(element)

        response = generate(
            tokenizer,
            model,
            sample.prompt,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=650,
        )[0]

        samples_for_fine_tuning_improvement_evaluation.append(sample.with_new_profile(response))

    # TODO unload the model from the GPU
    # TODO other evaluation model -> should be a large strong model
    # TODO proper tokenizer for the evaluation model

    number_of_wins_current_model = 0

    for sample in samples_for_fine_tuning_improvement_evaluation:
        if evaluate_is_profile1_preferred(
            model,
            tokenizer,
            Profile.parse(sample.best_profile_from_last_model),
            Profile.parse(sample.best_profile_from_original_model),
            sample.abstracts,
        ):
            number_of_wins_current_model += 1

    # Make a backup of the old file
    dump_json(
        load_json(SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE),
        SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE + START_DATETIME + '.old',
    )
    dump_json(
        samples_for_fine_tuning_improvement_evaluation,
        SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE,
    )

    # Return whether the current model is preferred more than the last model
    return number_of_wins_current_model > len(samples_for_fine_tuning_improvement_evaluation) * 0.5


if __name__ == '__main__':
    # Load all the database datasets into one Dataset
    with progress_status('Loading datasets'):
        files = [get_preference_output_file_path(START_DATETIME, i) for i in range(NUM_THREADS_EVALUATE)]
        train_dataset, test_dataset = load_dataset(files, tokenizer)

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

    # print sample from the dataset
    print('Sample from the dataset:')
    print('-' * 80)
    print(train_dataset[0])
    print('-' * 80)
    print(train_dataset[1])
    print('-' * 80)

    with progress_status('Loading model'):
        model = get_model()

    with progress_status('Loading trainer'):
        trainer = get_trainer(model)

    trainer.train()

    # save model at the end of training
    trainer.save_model()

    # free the memory again
    del model
    del trainer
    cuda.empty_cache()

    if not evaluate_model():
        print('The current model is preferred less than the last model. Should keep the last model.')
        sys.exit(1)  # exit with non-zero exit code to stop the iteration

    merge_and_save_model()
