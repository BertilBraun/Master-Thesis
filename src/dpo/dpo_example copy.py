from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
args, training_args, model_config = parser.parse_args_and_config()

################
# Model & Tokenizer
################
torch_dtype = (
    model_config.torch_dtype if model_config.torch_dtype in ['auto', None] else getattr(torch, model_config.torch_dtype)
)
quantization_config = get_quantization_config(model_config)
model_kwargs = dict(
    revision=model_config.model_revision,
    trust_remote_code=model_config.trust_remote_code,
    attn_implementation=model_config.attn_implementation,
    torch_dtype=torch_dtype,
    use_cache=False if training_args.gradient_checkpointing else True,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
    quantization_config=quantization_config,
)
model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
peft_config = get_peft_config(model_config)
if peft_config is None:
    model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
else:
    model_ref = None
tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
if args.ignore_bias_buffers:
    # torch distributed hack
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]
