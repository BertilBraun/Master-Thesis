from __future__ import annotations

"""
import torch
from datasets import Dataset
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


Dataset Format:

prompt: list[str]
chosen: list[str]
rejected: list[str]


Model Format:

model = AutoModelForCausalLM...
model_ref = AutoModelForCausalLM...

tokenizer = AutoTokenizer...


Load dataset:

ds = Dataset.from_dict({
    "prompt": prompt,
    "chosen": chosen,
    "rejected": rejected,
})

def process(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row

ds = ds.map(
    process,
    num_proc=multiprocessing.cpu_count(),
    load_from_cache_file=False,
)
    
train_dataset = ds["train"]
eval_dataset = ds["validation"]


Train:

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1, # the temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()




Train with PEFT:

# Load the base model.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/mixtral-8x7b-v0.1",
    load_in_4bit=True,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False

# Load the adapter.
model = PeftModel.from_pretrained(
    model,
    "/path/to/peft",
    is_trainable=True,
    adapter_name="train",
)
# Load the adapter a second time, with a different name, which will be our reference model.
model.load_adapter("/path/to/peft", adapter_name="reference")

# Initialize the trainer, without a ref_model param.
dpo_trainer = DPOTrainer(
    model,
    ...
    model_adapter_name="train",
    ref_adapter_name="reference",
)


"""

import json
import requests

from src.types import Profile, Competency, RankingResult


class DPOData:
    def __init__(self):
        self.prompts: list[str] = []
        self.chosen: list[str] = []
        self.rejected: list[str] = []

    def add(self, prompt: str, chosen: str, rejected: str) -> None:
        self.prompts.append(prompt)
        self.chosen.append(chosen)
        self.rejected.append(rejected)

    def __add__(self, other: DPOData) -> DPOData:
        new = DPOData()
        new.prompts = self.prompts + other.prompts
        new.chosen = self.chosen + other.chosen
        new.rejected = self.rejected + other.rejected
        return new

    def get_dataset(self) -> dict:
        return {
            'prompt': self.prompts,
            'chosen': self.chosen,
            'rejected': self.rejected,
        }


def get_jsonbin(route: str) -> dict | list[dict]:
    print('Getting jsonbin data from', 'https://api.jsonbin.io/v3' + route)
    response = requests.get(
        f'https://api.jsonbin.io/v3{route}',
        headers={
            'X-Master-Key': '$2a$10$F4XWL9xhJ1HtdWLMfj8aDeH4wzcYvl1evcpiFJJWNa3RUt9eLn6dm',
            'Content-Type': 'application/json',
        },
        json={},
    )

    return json.loads(response.text)


def get_jsonbin_bins() -> list[str]:
    # loads all the uncategorized bin ids from the jsonbin api and returns them as a list
    bins: list[str] = []

    ten_bins = get_jsonbin('/c/uncategorized/bins')
    for bin in ten_bins:
        bins.append(bin['record'])

    last_requested = ''

    while last_requested != bins[-1] and len(ten_bins) == 10:
        last_requested = bins[-1]
        ten_bins = get_jsonbin(f'/c/uncategorized/bins/{last_requested}')
        for bin in ten_bins:
            bins.append(bin['record'])

    return bins


def get_bin_data(bin_id: str) -> dict:
    return get_jsonbin(f'/b/{bin_id}')['record']  # type: ignore


def get_evaluation_data() -> dict:
    # TODO save to file and dont always refetch
    bins = get_jsonbin_bins()
    print('Found', len(bins), 'bins', bins)

    data = DPOData()

    for bin_id in bins:
        bin_data = get_bin_data(bin_id)

        author: str = bin_data['author']
        titles: list[str] = bin_data['titles']

        print('Processing bin', bin_id, 'by', author, 'with titles', titles)

        prompt = 'Some prompt about the extraction of these profiles' + ' '.join(titles)  # TODO proper prompt

        profile_mapping = {
            int(key): Profile(
                domain=value['profile']['domain'],
                competencies=[
                    Competency(name=comp['name'], description=comp['description'])
                    for comp in value['profile']['competencies']
                ],
            )
            for key, value in bin_data['profiles'].items()
        }

        for preference in bin_data['preferences']:
            instances: list[int] = preference['instances']
            preferred_instance_index: int = preference['preferred_instance']

            if (
                len(instances) != 2
                or preferred_instance_index not in [0, 1]
                or any(instance not in profile_mapping for instance in instances)
            ):
                print('Invalid preference:', preference)
                continue

            preferred_profile = profile_mapping[instances[preferred_instance_index]]
            other_profile = profile_mapping[instances[1 - preferred_instance_index]]

            data.add(prompt, str(preferred_profile), str(other_profile))

    return data.get_dataset()


def get_something(evals: list[RankingResult]) -> dict:
    # TODO write to file after the extraction ranking
    # The assumption is, that all the determined ranking results are in the list evals
    data = DPOData()


from pprint import pprint

pprint(get_evaluation_data())
exit()


from math import ceil
import multiprocessing
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    # DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

MODEL = 'gpt2'
TEST_PERCENTAGE = 0.1

prompts = [
    'What is the meaning of life?',
    'What is the best way to cook a steak?',
    'What is the capital of France?',
]
chosen = [
    'The meaning of life is to be happy.',
    'The best way to cook a steak is to grill it.',
    'The capital of France is Paris.',
]
rejected = [
    'The meaning of life is to be sad.',
    'The best way to cook a steak is to boil it.',
    'The capital of France is Berlin.',
]

ds = Dataset.from_dict(
    {
        'prompt': prompts,
        'chosen': chosen,
        'rejected': rejected,
    }
)

# Take the first TEST_PERCENTAGE (rounded up) of the dataset as the test dataset
test_samples = ceil(len(ds) * TEST_PERCENTAGE)
test_dataset = ds.select(range(test_samples))
train_dataset = ds.select(range(test_samples, len(ds)))

print(ds, test_dataset, train_dataset)

exit()

model = AutoModelForCausalLM.from_pretrained(MODEL)
model_ref = AutoModelForCausalLM.from_pretrained(MODEL)

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def process(row):
    row['chosen'] = tokenizer.apply_chat_template(row['chosen'], tokenize=False)
    row['rejected'] = tokenizer.apply_chat_template(row['rejected'], tokenize=False)
    return row


ds = ds.map(
    process,
    num_proc=multiprocessing.cpu_count(),
    load_from_cache_file=False,
)

train_dataset = ds['train']
eval_dataset = ds['validation']

# training_args = DPOConfig(
#     output_dir='output',
#     overwrite_output_dir=True,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     num_train_epochs=1,
#     logging_dir='logs',
#     logging_steps=10,
#     save_steps=10,
#     save_total_limit=1,
#     evaluation_strategy='steps',
#     eval_steps=10,
#     report_to='none',
#     logging_first_step=True,
#     load_best_model_at_end=True,
#     metric_for_best_model='eval_loss',
#     greater_is_better=False,
#     report_to='none',
# )

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    # args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
