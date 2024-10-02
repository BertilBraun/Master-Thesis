import os
from src.util.log import datetime_str
from src.util import write_to_file


NUM_SAMPLES_TO_GENERATE = 290  # Should be doable in 30 minutes on 4 GPUs

PAPERS_PER_SAMPLE = 4
TOP_K_TO_SAMPLE = 8
TEMPERATURE = 0.8  # Prefer more diverse samples so that all TOP_K are different
NUM_EXAMPLES = 1

EVALUATION_BATCH_SIZE = 8


# WARNING there is a copy of this variable in src/finetuning/train.py
OUTPUT_DIR = 'dpo_output'


# WARNING there is a copy of this variable in src/finetuning/train.py
CURRENT_MODEL_PATH = f'./{OUTPUT_DIR}/current-finetuned-model'
SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE = (
    f'{OUTPUT_DIR}/samples_for_fine_tuning_improvement_evaluation.json'
)

EVALUATION_MODEL_ID = 'meta-llama/Meta-Llama-3-70B-Instruct'
# WARNING there is a copy of this variable in src/finetuning/train.py
BASE_MODEL_ID = 'microsoft/Phi-3-mini-128k-instruct'

USE_FLASH_ATTENTION_FOR_EVALUATION = True

NUMBER_OF_SAMPLES_TO_EVALUATE_THE_IMPROVEMENT_ON_AFTER_TRAINING = 50


# TODO test parameters - comment out for production
# ---------------------------------------

# EVALUATION_MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'
# USE_FLASH_ATTENTION_FOR_EVALUATION = False

# EVALUATION_BATCH_SIZE = 4

# NUM_SAMPLES_TO_GENERATE = 32

# ---------------------------------------


def get_new_datetime_str() -> str:
    start_datetime = datetime_str()
    write_to_file(f'{OUTPUT_DIR}/start_datetime.txt', start_datetime)
    return start_datetime


def get_previous_datetime_str() -> str:
    # WARNING there is a copy of this function in src/finetuning/train.py
    assert os.path.exists(
        f'{OUTPUT_DIR}/start_datetime.txt'
    ), 'Run get_new_datetime_str() first to create the start_datetime.txt file.'
    with open(f'{OUTPUT_DIR}/start_datetime.txt', 'r') as f:
        return f.read()


def get_profile_output_file_path(start_datetime: str) -> str:
    return f'{OUTPUT_DIR}/samples_to_evaluate/{start_datetime}.json'


def get_preference_output_file_path(start_datetime: str) -> str:
    # WARNING there is a copy of this function in src/finetuning/train.py
    return f'{OUTPUT_DIR}/preferences/{start_datetime}.json'
