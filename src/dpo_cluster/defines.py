import os
from torch import Tensor, float16, compile
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)
from dataclasses import dataclass
from src.types import Message
from src.log import datetime_str
from src.util import write_to_file


"""
how many samples to we generate with each extraction of TOP_K_TO_SAMPLE?
 - 4 papers per sample
 - TOP_K_TO_SAMPLE extracted profiles
 - TOP_K_TO_SAMPLE profiles in a tournament
 - TOP_K_TO_SAMPLE - 1 comparisons in a tournament
 - TOP_K_TO_SAMPLE = 16 -> 32 usable preferences and 15 comparisons
 - TOP_K_TO_SAMPLE = 8 -> 12 usable preferences and 7 comparisons
=> higher TOP_K_TO_SAMPLE means more usable preferences with comparativly less comparisons
   but limited by the number of good profiles we can extract with such a high TEMPERATURE

TODO are the TOP_K_TO_SAMPLE samples different enough?

TODO how long does extracting NUM_SAMPLES_TO_GENERATE samples take? Measure it!
Theoretically:
 - NUM_SAMPLES_TO_GENERATE samples / 32 preferences = 63 tournaments
 - 63 tournaments * 15 comparisons = 945 comparisons
 - 945 comparisons * 30 seconds / NUM_THREADS_EVALUATE = 1.6 hours
 - 63 extractions * 30 seconds * TOP_K_TO_SAMPLE / NUM_THREADS_GENERATE = 2.8 hours
TODO how do generating and evaluating compare in time? Do we need more threads for one or the other?

How much would 10k training samples cost?
  - Approximately 3.0k Tokens in a one-shot prompt
  - ~300 tokens for the response
  - 1M tokens input = 5$
  - 1M tokens output = 15$
  - 945 comparisons * 3.0k tokens = 2.8M tokens => 2.8M tokens * 5$/1M tokens = 14$ for input
  - 945 comparisons * 300 tokens = 283.5k tokens => 283.5k tokens * 15$/1M tokens = 4.25$ for output
  - ~19$ per 2000 Samples
  - Can be cut to 14$ with 1x batching 
    - 19$ * 3/4 = ~14$ 
    - since half of the comparisons are in the first round and these would be batched with half the price
    - 1 day waiting
  - Can be cut to 12$ with 2x batching 
    - 19$ * 5/8 = ~12$
    - batching the first round and then also the second round
    - 2 days waiting

"""

NUM_SAMPLES_TO_GENERATE = 2000  # TODO less? more?

PAPERS_PER_SAMPLE = 4
TOP_K_TO_SAMPLE = 16
TEMPERATURE = 1.2  # Prefer more diverse samples so that all TOP_K are different
NUM_EXAMPLES = 1  # TODO or 0?

NUM_THREADS_GENERATE = 3
NUM_THREADS_EVALUATE = 5


TEST_PERCENTAGE = 0.05
OUTPUT_DIR = 'dpo_output'
TRAINING_OUTPUT_DIR = f'{OUTPUT_DIR}/training'

EVALUATION_MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'
BASE_MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'  # TODO tbd
CURRENT_MODEL_PATH = f'./{OUTPUT_DIR}/current-finetuned-model'

NUMBER_OF_EPOCHS_TO_TRAIN = 3
NUMBER_OF_SAMPLES_TO_EVALUATE_THE_IMPROVEMENT_ON_AFTER_TRAINING = 50
SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE = (
    f'{OUTPUT_DIR}/samples_for_fine_tuning_improvement_evaluation.json'
)

# TODO test parameters - comment out for production
# ---------------------------------------

NUM_SAMPLES_TO_GENERATE = 32

PAPERS_PER_SAMPLE = 4
TOP_K_TO_SAMPLE = 4

NUM_THREADS_GENERATE = 1
NUM_THREADS_EVALUATE = 1

# ---------------------------------------


TEST_PERCENTAGE = 1 / NUM_SAMPLES_TO_GENERATE  # only one test sample
OUTPUT_DIR = 'dpo_output_test'
TRAINING_OUTPUT_DIR = f'{OUTPUT_DIR}/training'

CURRENT_MODEL_PATH = f'./{OUTPUT_DIR}/current-finetuned-model'

NUMBER_OF_EPOCHS_TO_TRAIN = 2
NUMBER_OF_SAMPLES_TO_EVALUATE_THE_IMPROVEMENT_ON_AFTER_TRAINING = 5
SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE = (
    f'{OUTPUT_DIR}/samples_for_fine_tuning_improvement_evaluation.json'
)


def get_new_datetime_str() -> str:
    start_datetime = datetime_str()
    write_to_file(f'{OUTPUT_DIR}/start_datetime.txt', start_datetime)
    return start_datetime


def get_previous_datetime_str() -> str:
    assert os.path.exists(
        f'{OUTPUT_DIR}/start_datetime.txt'
    ), 'Run get_new_datetime_str() first to create the start_datetime.txt file.'
    with open(f'{OUTPUT_DIR}/start_datetime.txt', 'r') as f:
        return f.read()


@dataclass(frozen=True)
class SampleForFineTuningImprovementEvaluation:
    prompt: str
    abstracts: list[str]
    best_profile_from_original_model: str
    best_profile_from_last_model: str

    @staticmethod
    def from_json(data: dict) -> 'SampleForFineTuningImprovementEvaluation':
        return SampleForFineTuningImprovementEvaluation(
            prompt=data['prompt'],
            abstracts=data['abstracts'],
            best_profile_from_original_model=data['best_profile_from_original_model'],
            best_profile_from_last_model=data['best_profile_from_last_model'],
        )

    def with_new_profile(self, best_profile_from_last_model: str) -> 'SampleForFineTuningImprovementEvaluation':
        return SampleForFineTuningImprovementEvaluation(
            prompt=self.prompt,
            abstracts=self.abstracts,
            best_profile_from_original_model=self.best_profile_from_original_model,
            best_profile_from_last_model=best_profile_from_last_model,
        )


def get_tokenizer(name_or_path: str = BASE_MODEL_ID) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"

    tokenizer.padding_side = 'left'  # to prevent errors with FA
    tokenizer.truncation_side = 'left'  # to prevent cutting off last generation

    return tokenizer


def get_model(
    name_or_path: str = CURRENT_MODEL_PATH,
    device='cuda',
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=float16 if not load_in_4bit and not load_in_8bit else None,
        device_map=device,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
    )
    model = model.eval()
    model.generation_config.cache_implementation = 'sdpa'

    compiled_model = compile(model, mode='reduce-overhead', fullgraph=True)

    return compiled_model  # type: ignore


def prompt_messages_to_str(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, messages: list[Message]) -> str:
    return tokenizer.apply_chat_template(conversation=[message.to_dict() for message in messages], tokenize=False)  # type: ignore


def clean_output(output: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> str:
    output = output.strip()
    output = output.replace(tokenizer.eos_token, '')

    output = output.strip()
    if output.startswith('assistant'):
        output = output[len('assistant') :]

    output = output.strip()
    if output.startswith(':'):
        output = output[1:]

    output = output.strip()
    if not output.startswith('{'):
        output = '{' + output

    return output


def generate(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model: PreTrainedModel,
    prompt: str,
    /,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    max_new_tokens: int = 300,
    temperature: float = 0.2,
    skip_special_tokens: bool = True,
) -> list[str]:
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,  # TODO remove?
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids('<|eot_id|>'),
    ]  # TODO specifically for Meta-Llama3-8B-Instruct

    outputs: Tensor = model.generate(
        **inputs,  # type: ignore
        num_return_sequences=num_return_sequences,
        num_beams=num_return_sequences,
        num_beam_groups=num_return_sequences,
        do_sample=do_sample if num_return_sequences == 1 else False,
        diversity_penalty=temperature if num_return_sequences > 1 else 0.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        temperature=temperature if do_sample and num_return_sequences == 1 else None,
        top_p=0.8,
    )

    input_length = inputs.input_ids.shape[1]

    # TODO really intricate logging -> check that all tokens are correct (including EOS token, etc.)

    output_strs = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=skip_special_tokens)

    return [clean_output(output_str, tokenizer) for output_str in output_strs]
