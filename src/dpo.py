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
import sqlite3

from enum import Enum
from typing import Dict, List

from src.types import AuthorResult, Profile, Competency
from src.papers import get_paper_by_title
from src.log import log, LogLevel


class EvaluationType(Enum):
    EXPERT = 'expert'
    AUTOMATIC = 'automatic'


class DPODatabase:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup()

    def setup(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                chosen TEXT,
                rejected TEXT,
                evaluation_type TEXT,
                external_id OPTIONAL TEXT,
                author_name TEXT
            )
        """
        )
        self.conn.commit()

    def add_entry(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        eval_type: EvaluationType,
        author_name: str,
        external_id: str | None = None,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO preferences (prompt, chosen, rejected, evaluation_type, external_id, author_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (prompt, chosen, rejected, eval_type.value, external_id, author_name),
        )
        self.conn.commit()

    def get_entries_by_type(self, eval_type: EvaluationType) -> Dict[str, List[str]]:
        self.cursor.execute(
            """
            SELECT prompt, chosen, rejected FROM preferences WHERE evaluation_type=?
        """,
            (eval_type.value,),
        )
        rows = self.cursor.fetchall()

        data = {'prompt': [], 'chosen': [], 'rejected': []}
        for prompt, chosen, rejected in rows:
            data['prompt'].append(prompt)
            data['chosen'].append(chosen)
            data['rejected'].append(rejected)

        return data

    def check_existence_by_external_id(self, external_id: str) -> bool:
        self.cursor.execute(
            """
            SELECT 1 FROM preferences WHERE external_id=?
        """,
            (external_id,),
        )
        exists = self.cursor.fetchone() is not None
        return exists

    def check_existence_by_author_name_and_eval_type(self, author_name: str, eval_type: EvaluationType) -> bool:
        self.cursor.execute(
            """
            SELECT 1 FROM preferences WHERE author_name=? AND evaluation_type=?
        """,
            (author_name, eval_type.value),
        )
        exists = self.cursor.fetchone() is not None
        return exists


class JsonBin:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get(self, route: str) -> dict | list[dict]:
        print('Getting jsonbin data from', 'https://api.jsonbin.io/v3' + route)
        response = requests.get(
            f'https://api.jsonbin.io/v3{route}',
            headers={
                'X-Master-Key': self.api_key,
                'Content-Type': 'application/json',
            },
            json={},
        )

        return json.loads(response.text)

    def bins(self) -> list[str]:
        # loads all the uncategorized bin ids from the jsonbin api and returns them as a list
        bins: list[str] = []

        ten_bins = self.get('/c/uncategorized/bins')
        for bin in ten_bins:
            bins.append(bin['record'])

        last_requested = ''

        while last_requested != bins[-1] and len(ten_bins) == 10:
            last_requested = bins[-1]
            ten_bins = self.get(f'/c/uncategorized/bins/{last_requested}')
            for bin in ten_bins:
                bins.append(bin['record'])

        return bins

    def bin(self, bin_id: str) -> dict:
        return self.get(f'/b/{bin_id}')['record']  # type: ignore


def get_dataset_from_expert_evaluation(db: DPODatabase) -> None:
    jsonbin = JsonBin(api_key='$2a$10$F4XWL9xhJ1HtdWLMfj8aDeH4wzcYvl1evcpiFJJWNa3RUt9eLn6dm')
    bins = jsonbin.bins()
    print('Found', len(bins), 'bins', bins)

    for bin_id in bins:
        if db.check_existence_by_external_id(bin_id):
            print('Bin', bin_id, 'already exists in the database')
            continue

        bin_data = jsonbin.bin(bin_id)

        author: str = bin_data['author']
        titles: list[str] = bin_data['titles']
        print('Processing bin', bin_id, 'by', author, 'with titles', titles)

        papers = [get_paper_by_title(title) for title in titles]
        abstracts = [paper.abstracts[0] for paper in papers if paper]

        prompt = 'Some prompt about the extraction of these profiles\n\n\n' + '\n\n'.join(
            abstracts
        )  # TODO proper prompt

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

            db.add_entry(prompt, str(preferred_profile), str(other_profile), EvaluationType.EXPERT, author, bin_id)


def get_dataset_from_automatic_evaluation(db: DPODatabase, evaluation: AuthorResult) -> None:
    # The assumption is, that all the determined ranking results are in the list preferences

    if db.check_existence_by_author_name_and_eval_type(evaluation.author, EvaluationType.AUTOMATIC):
        print('Automatic evaluation by', evaluation.author, 'already exists in the database')
        return

    papers = [get_paper_by_title(title) for title in evaluation.titles]
    abstracts = [paper.abstracts[0] for paper in papers if paper]

    prompt = 'Some prompt about the extraction of these profiles\n\n\n' + '\n\n'.join(abstracts)  # TODO proper prompt

    for preference in evaluation.preferences:
        preferred_profile = preference.winner.profile
        other_profile = preference.loser.profile

        db.add_entry(prompt, str(preferred_profile), str(other_profile), EvaluationType.AUTOMATIC, evaluation.author)


db = DPODatabase('dpo.db')

# get_dataset_from_automatic_evaluation(db, AuthorResult(
#     author='Test Author',
#     titles=['Test Title 1', 'Test Title 2'],
#     root=TournamentNode(),
#     preferences=[
#         RankingResult(
#             profiles=(Profile('Test Domain 1', [Competency('Test Competency 1', 'Test Description 1')]),
#                       Profile('Test Domain 2', [Competency('Test Competency 2', 'Test Description 2')])),
#             preferred_profile=0,
#             reasoning='Test Reasoning 1',
#         ),
#     ],
# )

get_dataset_from_expert_evaluation(db)

log('Expert data:')
log(db.get_entries_by_type(EvaluationType.EXPERT), use_pprint=True)
log('Automatic data:')
log(db.get_entries_by_type(EvaluationType.AUTOMATIC), use_pprint=True)

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
