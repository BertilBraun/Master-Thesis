import gc
import os
from torch import Tensor, bfloat16, compile, cuda
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    BitsAndBytesConfig,
)
from src.logic.types import Message

from src.finetuning.defines import CURRENT_MODEL_PATH
from src.util.cache import generate_hashcode
from src.util.json import dump_json, load_json


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
        # bnb_4bit_compute_dtype=float16 if not quantized else float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype='auto',
        device_map=device,
        quantization_config=bnb_config if quantized else None,
        trust_remote_code=True,
        attn_implementation='flash_attention_2' if use_flash_attention else None,
        low_cpu_mem_usage=True,
        # local_files_only=name_or_path.startswith('./'),
        # NOTE: This is a hack which seems to work for Phi3 to avoid a bug when the context size crosses 4096 tokens during generation
        # original_max_position_embeddings=8192 * 2,
    )
    model = model.eval()

    compiled_model = compile(model, mode='reduce-overhead', fullgraph=True)

    return compiled_model  # type: ignore


def prompt_messages_to_str(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, messages: list[Message]) -> str:
    prompt: str = tokenizer.apply_chat_template(
        conversation=[message.to_dict() for message in messages],  # type: ignore
        tokenize=False,
    )

    return prompt
    prompt = prompt.replace(tokenizer.eos_token, '').strip()
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
    cache_folder = 'finetuning/logic/LLM_output'
    os.makedirs(cache_folder, exist_ok=True)

    hash_code = generate_hashcode(
        (model.name_or_path, prompt, num_return_sequences, do_sample, max_new_tokens, temperature, skip_special_tokens)
    )
    cache_file_name = os.path.join(cache_folder, hash_code + '.json')

    if os.path.exists(cache_file_name):
        return load_json(cache_file_name)

    inputs = tokenizer(tokenizer.eos_token + prompt, return_tensors='pt', padding=True).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        # The following are phi3 specific tokens
        # 2,
        # 32000,
        # 32007,
        # 32010,
    ]

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
            f.write(f'\n\n\n\n{clean_output(output_str, tokenizer)}\n\n\n\n' + '-' * 100)
        f.write('\n\n\n\n' + '=' * 100 + '\n\n\n\n')

    dump_json(output_strs, cache_file_name)

    gc.collect()
    cuda.empty_cache()

    return [clean_output(output_str, tokenizer) for output_str in output_strs]
