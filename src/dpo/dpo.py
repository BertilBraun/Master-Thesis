import multiprocessing

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TrainingArguments,
)
from trl import (
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from src.dpo.dpo_database import DPODatabase, EvaluationType
from src.log import progress_status

MODEL = 'gpt2'
TEST_PERCENTAGE = 0.05
OUTPUT_DIR = 'dpo_output'


def load_dataset(
    db: DPODatabase,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    evaluation_type: EvaluationType,
    test_percentage: float = TEST_PERCENTAGE,
) -> tuple[Dataset, Dataset]:
    """
    # TODO add the doc string with copilot :D
    """

    # Dataset with:
    # prompt: list[str]
    # chosen: list[str]
    # rejected: list[str]
    ds = Dataset.from_dict(db.get_entries_by_type(evaluation_type))

    def process(row):
        row['chosen'] = tokenizer.apply_chat_template(row['chosen'], tokenize=False)
        row['rejected'] = tokenizer.apply_chat_template(row['rejected'], tokenize=False)
        return row

    ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

    ds = ds.train_test_split(test_size=test_percentage, shuffle=True)

    return ds['train'], ds['validation']


def get_tokenizer(name: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"

    return tokenizer


db = DPODatabase('dpo.db')

# add_to_dataset_from_expert_evaluation(db)
# add_to_dataset_from_automatic_evaluation(db, ...)

# log('Expert data:')
# log(db.get_entries_by_type(EvaluationType.EXPERT), use_pprint=True)
# log('Automatic data:')
# log(db.get_entries_by_type(EvaluationType.AUTOMATIC), use_pprint=True)

model = AutoModelForCausalLM.from_pretrained(MODEL)
model_ref = AutoModelForCausalLM.from_pretrained(MODEL)

tokenizer = get_tokenizer(MODEL)

train_dataset, test_dataset = load_dataset(db, tokenizer, EvaluationType.AUTOMATIC)

print(train_dataset, test_dataset)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    logging_dir='logs',
    logging_steps=10,
    save_steps=10,
    save_total_limit=1,
    evaluation_strategy='steps',
    eval_steps=10,
    report_to='none',
    logging_first_step=True,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

with progress_status('Initializing the DPOTrainer...'):
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,  # the temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
        callbacks=[RichProgressCallback()],
    )

dpo_trainer.train()


with progress_status('Training completed! Saving the model to ' + training_args.output_dir):
    dpo_trainer.save_model(training_args.output_dir)


# Setup the initial model which will be fine-tuned using the DPO Algorithm

while True:
    # Generate Samples (Using the current model, generate 2 samples (higher temperature) for each prompt which will later be ranked)
    # Generate more than 2 samples and get tournament like preferences between them?
    # run_query_for_instance()? -> Beam search and return the top k samples?

    # Rank them (Using a strong baseline model like GPT-4)
    # tournament_ranking()? src.openai_defines.BASE_URL_LLM = None?

    # Fine-tune the model using the DPO Algorithm

    # Repeat
    if True:  # Check for convergence condition here
        break


"""
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

"""
Possible good default settings:

# peft:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""
