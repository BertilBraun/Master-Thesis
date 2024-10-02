from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from src.finetuning.defines import BASE_MODEL_ID


def get_tokenizer(name_or_path: str = BASE_MODEL_ID) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    # WARNING there is a copy of this function in src/finetuning/train.py
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
