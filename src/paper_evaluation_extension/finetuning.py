import os
import gc
import sys
import random
from math import ceil, log2
from typing import Callable, Sequence, Type, TypeVar

import torch
import huggingface_hub
import multiprocessing
from torch import cuda, float16
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from trl import DPOTrainer, DPOConfig

from src.logic.types.instance_type import Query
from src.logic.types.database_types import EvaluationResult
from src.finetuning.logic.finetuning_types import (
    SampleForFineTuningImprovementEvaluation,
    SampleToEvaluate,
    SampleToGenerate,
    PreferenceSample,
)
from src.logic.database import get_retriever_getter
from src.logic.papers import get_random_english_authors_abstracts
from src.extraction.extraction_custom import prompt_for_extract_from_abstracts_custom
from src.extraction.evaluation import (
    get_all_preferences,
    prompt_for_ranking,
    run_tournament_ranking,
    default_round_evaluator,
)
from src.logic.types import (
    Example,
    Profile,
    EvaluationResult_from_invalid_response,
    Ranking,
)
from src.util import dump_json, ratio, log_all_exceptions, timeblock, FromJsonProtocol, load_json
from src.finetuning.logic.tokenizer import get_tokenizer
from src.finetuning.logic.model import generate, get_model, prompt_messages_to_str

print('All imports done. Now starting execution...')


NUM_SAMPLES_TO_GENERATE = 500

PAPERS_PER_SAMPLE = 4
TOP_K_TO_SAMPLE = 8
TEMPERATURE = 0.8  # Prefer more diverse samples so that all TOP_K are different
NUM_EXAMPLES = 1

OUTPUT_DIR = 'dpo_output'

TEST_PERCENTAGE = 0.002

TRAINING_OUTPUT_DIR = f'{OUTPUT_DIR}/training'

NUMBER_OF_EPOCHS_TO_TRAIN = 6


CURRENT_MODEL_PATH = f'./{OUTPUT_DIR}/current-finetuned-model'
SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE = (
    f'{OUTPUT_DIR}/samples_for_fine_tuning_improvement_evaluation.json'
)

EVALUATION_MODEL_ID = 'meta-llama/Llama-3.1-70B-Instruct'
BASE_MODEL_ID = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

NUMBER_OF_SAMPLES_TO_EVALUATE_THE_IMPROVEMENT_ON_AFTER_TRAINING = 30


# TODO temporary for testing
NUMBER_OF_EPOCHS_TO_TRAIN = 1
NUMBER_OF_SAMPLES_TO_EVALUATE_THE_IMPROVEMENT_ON_AFTER_TRAINING = 1
NUM_SAMPLES_TO_GENERATE = 12
TOP_K_TO_SAMPLE = 8
PAPERS_PER_SAMPLE = 2
TEMPERATURE = 0.8
TEST_PERCENTAGE = 1 / 12


def setup_initial_model(model_path: str, base_model_id: str) -> bool:
    """Setup the initial model for finetuning."""
    if os.path.exists(model_path):
        print(f'{model_path} already exists. Exiting...')
        return False
    # Initialize and save the base model
    model = get_model(base_model_id)
    model.save_pretrained(model_path)
    return True


def generate_evaluation_samples(model_path: str) -> list[SampleForFineTuningImprovementEvaluation]:
    """Generate initial samples for evaluating model improvement."""

    def process_queries(queries: list[Query], device_id: int) -> list[SampleForFineTuningImprovementEvaluation]:
        tokenizer = get_tokenizer()
        model = get_model(
            model_path,
            device=f'cuda:{device_id}',
            load_in_8bit=True,
        )

        samples: list[SampleForFineTuningImprovementEvaluation] = []

        for query in queries:
            abstracts = '\n\n'.join(query.abstracts)
            examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Example).invoke(abstracts)
            prompt_messages = prompt_for_extract_from_abstracts_custom(query.abstracts, examples)
            prompt = prompt_messages_to_str(tokenizer, prompt_messages)
            prompt += 'Domain: "'  # Note: This is a hack to start the prompt with the desired format to encourage the model to generate valid profiles
            response = generate(
                tokenizer,
                model,
                prompt,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=650,
            )[0]
            profile = Profile.parse('Domain: "' + response)
            samples.append(
                SampleForFineTuningImprovementEvaluation(
                    prompt=prompt,
                    abstracts=query.abstracts,
                    best_profile_from_original_model=str(profile),
                    best_profile_from_last_model=str(profile),
                )
            )

        return samples

    author_queries = get_random_english_authors_abstracts(
        NUMBER_OF_SAMPLES_TO_EVALUATE_THE_IMPROVEMENT_ON_AFTER_TRAINING, PAPERS_PER_SAMPLE
    )
    return map_over_devices(
        process_queries,
        author_queries,
        Query,
        SampleForFineTuningImprovementEvaluation,
    )


def load_dataset(preferences: list[PreferenceSample], test_percentage: float = 0.002) -> tuple[Dataset, Dataset]:
    """Load and split the dataset into training and test sets."""
    prompts = [sample.prompt for sample in preferences]
    chosens = [sample.chosen for sample in preferences]
    rejecteds = [sample.rejected for sample in preferences]
    dataset = Dataset.from_dict(
        {
            'prompt': prompts,
            'chosen': chosens,
            'rejected': rejecteds,
        }
    )
    split_dataset = dataset.train_test_split(test_size=test_percentage, shuffle=True)
    return split_dataset['train'], split_dataset['test']


def get_trainer(
    model,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_seq_length: int,
    prompt_length: int,
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> DPOTrainer:
    """Create a DPO trainer for model finetuning."""
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


def merge_and_save_model(next_model_path: str):
    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        TRAINING_OUTPUT_DIR,
        torch_dtype=float16,
        low_cpu_mem_usage=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(next_model_path, safe_serialization=True, max_shard_size='2GB')


def train_model(model_path: str, next_model_path: str, preferences: list[PreferenceSample]):
    """Train the finetuned model on the preference dataset."""
    train_dataset, test_dataset = load_dataset(preferences)
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

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        use_cache=False,
        attn_implementation='flash_attention_2',
        torch_dtype='auto',
        load_in_8bit=True,
        original_max_position_embeddings=8192 * 2,
    )
    trainer = get_trainer(
        model,
        tokenizer,
        max_seq_length,
        prompt_length,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )
    trainer.train()
    trainer.save_model()

    # free the memory again
    del model
    del trainer

    gc.collect()
    cuda.empty_cache()

    merge_and_save_model(next_model_path)


def evaluate_is_profile1_preferred(
    profile1: Profile,
    profile2: Profile,
    abstracts: list[str],
    model,
    tokenizer,
) -> bool:
    profiles = [profile1, profile2]
    random.shuffle(profiles)

    examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking).invoke('\n\n'.join(abstracts))
    prompt = prompt_for_ranking(profiles[0], profiles[1], examples, abstracts)

    response = generate(
        tokenizer,
        model,
        prompt_messages_to_str(tokenizer, prompt),
        num_return_sequences=1,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=650,
    )[0]

    evaluation = EvaluationResult_from_invalid_response(response)
    preferred_profile_index = Ranking.parse_preferred_profile_json(evaluation)

    return profiles[preferred_profile_index] == profile1


def get_wins_of_current_model(
    samples_to_evaluate: list[SampleForFineTuningImprovementEvaluation], device_id: int
) -> list[bool]:
    tokenizer = get_tokenizer(EVALUATION_MODEL_ID)
    model = get_model(
        EVALUATION_MODEL_ID,
        device=f'cuda:{device_id}',
        load_in_8bit=True,
    )

    return [
        evaluate_is_profile1_preferred(
            Profile.parse(sample.best_profile_from_last_model),
            Profile.parse(sample.best_profile_from_original_model),
            sample.abstracts,
            model,
            tokenizer,
        )
        for sample in tqdm(samples_to_evaluate, desc='Evaluating samples')
    ]


def evaluate_model(
    model_path: str, samples: list[SampleForFineTuningImprovementEvaluation]
) -> tuple[bool, list[SampleForFineTuningImprovementEvaluation]]:
    """Evaluate the finetuned model to assess improvements."""

    def process_samples(
        samples: list[SampleForFineTuningImprovementEvaluation], device_id: int
    ) -> list[SampleForFineTuningImprovementEvaluation]:
        tokenizer = get_tokenizer()
        model = get_model(
            model_path,
            device=f'cuda:{device_id}',
            load_in_8bit=True,
        )

        for sample in samples:
            response = generate(
                tokenizer,
                model,
                sample.prompt,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=650,
            )[0]
            new_profile = Profile.parse('Domain: "' + response)
            sample.best_profile_from_last_model = str(new_profile)

        del model
        gc.collect()
        cuda.empty_cache()

        return samples

    samples = map_over_devices(
        process_samples,
        samples,
        SampleForFineTuningImprovementEvaluation,
        SampleForFineTuningImprovementEvaluation,
    )

    with timeblock('Comparing the current model to the original model'):
        number_of_wins_current_model = sum(
            map_over_devices(
                get_wins_of_current_model,
                samples,
                SampleForFineTuningImprovementEvaluation,
                bool,
            )
        )

    total_samples = len(samples)
    print(f'The current model won {ratio(number_of_wins_current_model, total_samples)} against the original model')

    has_improved = number_of_wins_current_model / total_samples > 0.5

    return has_improved, samples


def calculate_number_of_authors_to_process(samples_to_generate: int, top_k: int) -> int:
    def calculate_P(n: int) -> int:
        log2_n = int(log2(n))
        sum_part = sum((2 ** (log2_n - r) * (2 ** (r - 1) - 1)) for r in range(1, log2_n + 1))
        return (n - 1) + sum_part

    return ceil(samples_to_generate / calculate_P(top_k))


def generate_samples(
    model_path: str,
    num_samples_to_generate: int,
    top_k: int = TOP_K_TO_SAMPLE,
) -> list[SampleToEvaluate]:
    """Generate new samples for finetuning."""
    samples_to_generate: list[SampleToGenerate] = []
    num_authors_to_process = calculate_number_of_authors_to_process(num_samples_to_generate, top_k)

    for query in get_random_english_authors_abstracts(num_authors_to_process, PAPERS_PER_SAMPLE):
        examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Example).invoke(
            '\n\n'.join(query.abstracts)
        )
        sample = SampleToGenerate(
            author=query.author,
            abstracts=query.abstracts,
            examples=examples,
        )
        samples_to_generate.append(sample)

    def process_samples_on_device(samples: list[SampleToGenerate], device_id: int) -> list[SampleToEvaluate]:
        torch.cuda.set_device(device_id)

        tokenizer = get_tokenizer()
        model = get_model(
            model_path,
            device=f'cuda:{device_id}',
            load_in_8bit=True,
        )

        results: list[SampleToEvaluate] = []

        for sample_to_generate in samples:
            prompt_messages = prompt_for_extract_from_abstracts_custom(
                sample_to_generate.abstracts,
                sample_to_generate.examples,
            )
            prompt = prompt_messages_to_str(tokenizer, prompt_messages)
            prompt += 'Domain: "'  # Note: This is a hack to start the prompt with the desired format to encourage the model to generate valid profiles
            responses = generate(
                tokenizer,
                model,
                prompt,
                num_return_sequences=top_k,
                temperature=TEMPERATURE,
                max_new_tokens=650,
            )

            profiles: list[Profile] = []
            for response in responses:
                with log_all_exceptions(f'Profile parsing failed for response: {response}'):
                    profiles.append(Profile.parse('Domain: "' + response))

            results.append(
                SampleToEvaluate(
                    author=sample_to_generate.author,
                    prompt=prompt,
                    abstracts=sample_to_generate.abstracts,
                    profiles=profiles,
                )
            )

        return results

    return map_over_devices(
        process_samples_on_device,
        samples_to_generate,
        SampleToGenerate,
        SampleToEvaluate,
    )


T = TypeVar('T', bound=FromJsonProtocol)
S = TypeVar('S', bound=FromJsonProtocol | bool | str | int | float)


def map_over_devices(
    func_to_apply: Callable[[list[T], int], Sequence[S]],
    all_elements: Sequence[T],
    obj_type: Type[T],
    return_type: Type[S],
) -> list[S]:
    num_devices = torch.cuda.device_count()
    elements_per_device = (len(all_elements) + num_devices - 1) // num_devices
    batches = [(i * elements_per_device, (i + 1) * elements_per_device) for i in range(num_devices)]

    def process_batch(args) -> None:
        (start, end), device_id = args
        torch.cuda.set_device(device_id)
        elements_batch = load_json('all_elements.json', obj_type)[start:end]
        result = func_to_apply(elements_batch, device_id)
        dump_json(result, f'result_{device_id}.json')

    dump_json(all_elements, 'all_elements.json')

    with multiprocessing.Pool(processes=num_devices) as pool:
        args = [(batches[i], i) for i in range(num_devices)]
        pool.map(process_batch, args)

    if return_type in [bool, str, int, float]:
        results = [load_json(f'result_{i}.json') for i in range(num_devices)]
    else:
        results = [
            load_json(
                f'result_{i}.json',
                return_type,  # type: ignore must be S
            )
            for i in range(num_devices)
        ]

    return [item for sublist in results for item in sublist]


def evaluate_samples(samples_to_evaluate: list[SampleToEvaluate]) -> list[PreferenceSample]:
    """Evaluate generated samples using a language model."""

    # TODO redo the evaluation system? Elo based - tradeof with computation time (n^2) - from 5 evaluations to 18 evaluations for no more preferences

    def process_samples_on_device(samples: list[SampleToEvaluate], device_id: int) -> list[PreferenceSample]:
        example_retriever = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking)

        tokenizer = get_tokenizer(EVALUATION_MODEL_ID)
        model = get_model(
            EVALUATION_MODEL_ID,
            device=f'cuda:{device_id}',
            load_in_8bit=True,
        )

        preference_samples: list[PreferenceSample] = []
        for sample_to_evaluate in samples:
            examples = example_retriever.invoke('\n\n'.join(sample_to_evaluate.abstracts))

            def match_evaluator(profile1_index: int, profile2_index: int) -> EvaluationResult:
                profiles = sample_to_evaluate.profiles
                prompt = prompt_for_ranking(
                    profiles[profile1_index],
                    profiles[profile2_index],
                    examples,
                    sample_to_evaluate.abstracts,
                )
                response = generate(
                    tokenizer,
                    model,
                    prompt_messages_to_str(tokenizer, prompt),
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=650,
                )[0]

                # TODO consistency check? Flip order and check if the result is the same?
                # TODO multiple models? Use different models and check if the result is the same?

                evaluation = EvaluationResult_from_invalid_response(response)
                return evaluation

            tournament = run_tournament_ranking(
                list(range(len(sample_to_evaluate.profiles))),
                default_round_evaluator(match_evaluator),
                do_shuffle=True,
            )

            preferences = []
            for preference in get_all_preferences(tournament):
                chosen = sample_to_evaluate.profiles[preference.winner]
                rejected = sample_to_evaluate.profiles[preference.loser]
                preferences.append(
                    PreferenceSample(
                        prompt=sample_to_evaluate.prompt,
                        chosen=str(chosen),
                        rejected=str(rejected),
                    )
                )
            preference_samples.extend(preferences)

        return preference_samples

    return map_over_devices(
        process_samples_on_device,
        samples_to_evaluate,
        SampleToEvaluate,
        PreferenceSample,
    )


def main():
    """Main function to coordinate finetuning steps."""
    huggingface_hub.login(new_session=False)

    print('Setting up initial model')
    start_model_path = f'{CURRENT_MODEL_PATH}_run_0'
    if setup_initial_model(start_model_path, BASE_MODEL_ID):
        # Setup returns True if the model was created
        print('Generating initial samples for evaluation')
        samples_for_improvement_evaluation = generate_evaluation_samples(start_model_path)
        dump_json(samples_for_improvement_evaluation, SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE)

    for i in range(10):
        model_path = f'{CURRENT_MODEL_PATH}_run_{i}'
        next_model_path = f'{CURRENT_MODEL_PATH}_run_{i + 1}'

        print(f'Finetuning model {i}')
        print(f'Generating samples for model {i}')
        generated_samples = generate_samples(model_path, NUM_SAMPLES_TO_GENERATE)
        dump_json(generated_samples, f'generated_samples_{i}.json')

        print(f'Evaluating samples for model {i}')
        preferences = evaluate_samples(generated_samples)
        dump_json(preferences, f'preferences_{i}.json')

        print(f'Training model {i}')
        train_model(model_path, next_model_path, preferences)

        print(f'Evaluating model {i}')
        has_improved, samples_for_improvement_evaluation = evaluate_model(
            next_model_path, samples_for_improvement_evaluation
        )
        dump_json(samples_for_improvement_evaluation, f'{SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE}_{i}')
        if not has_improved:
            print('Model has not improved. Exiting...')
            break

        print('Model has improved. Continuing to next iteration...')


if __name__ == '__main__':
    main()