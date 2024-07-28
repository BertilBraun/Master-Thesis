import gc
import os
from torch import Tensor, compile, cuda, float16, float32
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    BitsAndBytesConfig,
)
from dataclasses import dataclass
from src.types import Example, Message, Profile
from src.log import datetime_str
from src.util import write_to_file


CAS_OPENAI_API_KEY = 'sk-ce-service-account-OvzVRsc0DRXVJeCvxiQGT3BlbkFJmcquyYhxboiGGtFxshKi'

NUM_SAMPLES_TO_GENERATE = 320  # Should be doable in 30 minutes on 4 GPUs

PAPERS_PER_SAMPLE = 4
TOP_K_TO_SAMPLE = 8
TEMPERATURE = 0.8  # Prefer more diverse samples so that all TOP_K are different
NUM_EXAMPLES = 1

EVALUATION_BATCH_SIZE = 8


# WARNING there is a copy of this variable in src/dpo_cluster/train.py
OUTPUT_DIR = 'dpo_output'


# WARNING there is a copy of this variable in src/dpo_cluster/train.py
CURRENT_MODEL_PATH = f'./{OUTPUT_DIR}/current-finetuned-model'
SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE = (
    f'{OUTPUT_DIR}/samples_for_fine_tuning_improvement_evaluation.json'
)

EVALUATION_MODEL_ID = 'meta-llama/Meta-Llama-3-70B-Instruct'
# WARNING there is a copy of this variable in src/dpo_cluster/train.py
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
    # WARNING there is a copy of this function in src/dpo_cluster/train.py
    assert os.path.exists(
        f'{OUTPUT_DIR}/start_datetime.txt'
    ), 'Run get_new_datetime_str() first to create the start_datetime.txt file.'
    with open(f'{OUTPUT_DIR}/start_datetime.txt', 'r') as f:
        return f.read()


def get_profile_output_file_path(start_datetime: str) -> str:
    return f'{OUTPUT_DIR}/samples_to_evaluate/{start_datetime}.json'


def get_preference_output_file_path(start_datetime: str) -> str:
    # WARNING there is a copy of this function in src/dpo_cluster/train.py
    return f'{OUTPUT_DIR}/preferences/{start_datetime}.json'


@dataclass(frozen=True)
class SampleForFineTuningImprovementEvaluation:
    prompt: str
    abstracts: list[str]
    best_profile_from_original_model: str
    best_profile_from_last_model: str

    @staticmethod
    def from_json(data: dict) -> 'SampleForFineTuningImprovementEvaluation':
        return SampleForFineTuningImprovementEvaluation(**data)

    def with_new_profiles(
        self,
        best_profile_from_last_model: str,
        best_profile_from_original_model: str | None = None,
    ) -> 'SampleForFineTuningImprovementEvaluation':
        return SampleForFineTuningImprovementEvaluation(
            prompt=self.prompt,
            abstracts=self.abstracts,
            best_profile_from_original_model=best_profile_from_original_model or self.best_profile_from_original_model,
            best_profile_from_last_model=best_profile_from_last_model,
        )


@dataclass(frozen=True)
class SampleToGenerate:
    author: str
    abstracts: list[str]
    examples: list[Example]

    @staticmethod
    def from_json(data: dict) -> 'SampleToGenerate':
        return SampleToGenerate(
            author=data['author'],
            abstracts=data['abstracts'],
            examples=[Example.from_json(example) for example in data['examples']],
        )


@dataclass(frozen=True)
class SampleToEvaluate:
    author: str
    prompt: str
    abstracts: list[str]
    profiles: list[Profile]

    @staticmethod
    def from_json(data: dict) -> 'SampleToEvaluate':
        return SampleToEvaluate(
            author=data['author'],
            prompt=data['prompt'],
            abstracts=data['abstracts'],
            profiles=[Profile.from_json(profile) for profile in data['profiles']],
        )


@dataclass(frozen=True)
class PreferenceSample:
    # WARNING there is a copy of this class in src/dpo_cluster/train.py
    prompt: str
    chosen: str
    rejected: str

    @staticmethod
    def from_json(data: dict) -> 'PreferenceSample':
        return PreferenceSample(**data)


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

    return tokenizer


def get_model(
    name_or_path: str = CURRENT_MODEL_PATH,
    device='cuda',
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> PreTrainedModel:
    gc.collect()
    cuda.empty_cache()

    # use flash attention if the gpu is either A100 or H100
    use_flash_attention = any(name in cuda.get_device_name(0).lower() for name in ['a100', 'h100'])

    quantized = load_in_4bit or load_in_8bit

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=float16 if not quantized else float32,
    )

    print(f'Loading model from {name_or_path} on {device} with flash attention: {use_flash_attention}')
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype='auto',
        device_map=device,
        quantization_config=bnb_config if quantized else None,
        trust_remote_code=True,
        attn_implementation='flash_attention_2' if use_flash_attention else None,
        low_cpu_mem_usage=True,
        local_files_only=name_or_path.startswith('./'),
    )
    print(f'Model loaded from {name_or_path} on {device} with flash attention: {use_flash_attention}')
    print('Setting model to eval mode')
    model = model.eval()
    print('Model set to eval mode')

    return model
    compiled_model = compile(model, mode='reduce-overhead', fullgraph=True)

    return compiled_model  # type: ignore


def prompt_messages_to_str(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, messages: list[Message]) -> str:
    prompt: str = tokenizer.apply_chat_template(
        conversation=[message.to_dict() for message in messages],  # type: ignore
        tokenize=False,
    )
    prompt = prompt.replace('<|endoftext|>', '').strip() + '\n<|assistant|>'
    return prompt


def clean_output(output: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> str:
    output = output.strip()
    output = output.split('<|end|>')[0]
    output = output.replace(tokenizer.eos_token, '')

    output = output.strip()
    if output.startswith('assistant'):
        output = output[len('assistant') :]

    return output.strip()


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
    inputs = tokenizer(tokenizer.eos_token + prompt, return_tensors='pt', padding=True).to(model.device)

    terminators = [tokenizer.eos_token_id, tokenizer('<|end|>').input_ids[0], tokenizer('</s>').input_ids[0]]
    print(f'{terminators=}')

    do_sample = do_sample and num_return_sequences == 1

    outputs: Tensor = model.generate(
        **inputs,  # type: ignore
        num_return_sequences=num_return_sequences,
        num_beams=num_return_sequences,
        num_beam_groups=num_return_sequences,
        do_sample=do_sample,
        diversity_penalty=temperature if num_return_sequences > 1 else 0.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature if do_sample else None,
        top_p=0.8 if do_sample else None,
    )

    input_length = inputs.input_ids.shape[1]

    output_strs = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=skip_special_tokens)

    with open('LLM_output.txt', 'a') as f:
        f.write(f'\n\n\n\nPrompt: {prompt}\n\n\n\n')
        f.write('Outputs:\n\n')
        for output_str in output_strs:
            f.write(f'\n\n\n\n{output_str}\n\n\n\n' + '-' * 100)
        f.write('\n\n\n\n' + '=' * 100 + '\n\n\n\n')

    gc.collect()
    cuda.empty_cache()

    return [clean_output(output_str, tokenizer) for output_str in output_strs]


def batched_generate(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model: PreTrainedModel,
    prompts: list[str],
    /,
    do_sample: bool = True,
    max_new_tokens: int = 300,
    temperature: float = 0.2,
    skip_special_tokens: bool = True,
) -> list[str]:
    prompts = [tokenizer.eos_token + prompt for prompt in prompts]
    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)

    terminators = [tokenizer.eos_token_id, tokenizer('<|end|>').input_ids[0], tokenizer('</s>').input_ids[0]]

    outputs: Tensor = model.generate(
        **inputs,  # type: ignore
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature if do_sample else None,
        top_p=0.8 if do_sample else None,
    )

    input_length = inputs.input_ids.shape[1]

    output_strs = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=skip_special_tokens)

    with open('LLM_output.txt', 'a') as f:
        f.write(f'\n\n\n\nBatch Prompts: {prompts}\n\n\n\n')
        f.write('Outputs:\n\n')
        for output_str in output_strs:
            f.write(f'\n\n\n\n{output_str}\n\n\n\n' + '-' * 100)
        f.write('\n\n\n\n' + '=' * 100 + '\n\n\n\n')

    gc.collect()
    cuda.empty_cache()

    return [clean_output(output_str, tokenizer) for output_str in output_strs]
