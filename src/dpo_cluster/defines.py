from torch import float16
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
TEMPERATURE = 0.8  # Prefer more diverse samples so that all TOP_K are different
NUM_EXAMPLES = 1  # TODO or 0?

NUM_THREADS_GENERATE = 3
NUM_THREADS_EVALUATE = 5


TEST_PERCENTAGE = 0.05
OUTPUT_DIR = 'dpo_output'
TRAINING_OUTPUT_DIR = f'{OUTPUT_DIR}/training'

BASE_MODEL_ID = 'gpt2'  # TODO tbd
CURRENT_MODEL_PATH = f'./{OUTPUT_DIR}/current-finetuned-model'

NUMBER_OF_EPOCHS_TO_TRAIN = 3
NUMBER_OF_SAMPLES_TO_EVALUATE_THE_IMPROVEMENT_ON_AFTER_TRAINING = 50
SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE = (
    f'{OUTPUT_DIR}/samples_for_fine_tuning_improvement_evaluation.json'
)


def get_new_datetime_str() -> str:
    start_datetime = datetime_str()
    with open('dpo_output/start_datetime.txt', 'w') as f:
        f.write(start_datetime)
    return start_datetime


def get_previous_datetime_str() -> str:
    with open('dpo_output/start_datetime.txt', 'r') as f:
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"

    tokenizer.padding_side = 'left'  # to prevent errors with FA
    tokenizer.truncation_side = 'left'  # to prevent cutting off last generation
    return tokenizer


def get_model(name_or_path: str = CURRENT_MODEL_PATH, device='cuda') -> PreTrainedModel:
    # TODO allow to load in lower precision
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=float16)
    model = model.to(device)
    model = model.eval()

    return model


def prompt_messages_to_str(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, messages: list[Message]) -> str:
    return tokenizer.apply_chat_template(conversation=[message.to_dict() for message in messages], tokenize=False)  # type: ignore


def generate(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model: PreTrainedModel,
    prompt: str,
    /,
    num_return_sequences: int = 1,
    num_beams: int = 1,
    do_sample: bool = False,
    max_new_tokens: int = 300,
    temperature: float = 0.2,
    skip_special_tokens: bool = True,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)

    outputs: list[int] = model.generate(
        **inputs,  # type: ignore
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # TODO really intricate logging -> check that all tokens are correct (including EOS token, etc.)

    return tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)
