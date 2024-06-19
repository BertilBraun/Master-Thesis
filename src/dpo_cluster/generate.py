import itertools
from asyncio import run, sleep, gather
from queue import Queue

import torch
from peft.auto import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataclasses import dataclass

from src.log import datetime_str
from src.dpo.dpo_database import DPODatabase, EvaluationType
from src.evaluation import get_all_preferences
from src.database import format_example_messages, get_retriever_getter
from src.papers import get_random_english_authors_abstracts
from src.types import Example, HumanMessage, Message, Profile, SystemMessage, TournamentNode


NUM_SAMPLES_TO_GENERATE = 2000  # TODO less?

# how many samples to we generate with each extraction of TOP_K_TO_SAMPLE?
#      - 4 papers per sample
#      - TOP_K_TO_SAMPLE extracted profiles
#      - TOP_K_TO_SAMPLE profiles in a tournament
#      - TOP_K_TO_SAMPLE - 1 comparisons in a tournament
#      - TOP_K_TO_SAMPLE = 16 -> 32 usable preferences and 15 comparisons
#      - TOP_K_TO_SAMPLE = 8 -> 12 usable preferences and 7 comparisons
# => higher TOP_K_TO_SAMPLE means more usable preferences with comparativly less comparisons
#    but limited by the number of good profiles we can extract with such a high TEMPERATURE

# TODO how long does extracting NUM_SAMPLES_TO_GENERATE samples take?
#      - NUM_SAMPLES_TO_GENERATE samples / 32 preferences = 63 tournaments
#      - 63 tournaments * 15 comparisons = 945 comparisons
#      - 945 comparisons * 30 seconds / NUM_THREADS_EVALUATE = 1.6 hours
#      - 63 extractions * 30 seconds * TOP_K_TO_SAMPLE / NUM_THREADS_GENERATE = 2.8 hours
# TODO how do generating and evaluating compare in time? Do we need more threads for one or the other?

# TODO are the TOP_K_TO_SAMPLE samples different enough?

PAPERS_PER_SAMPLE = 4
TOP_K_TO_SAMPLE = 16
TEMPERATURE = 0.8  # Prefer more diverse samples so that all TOP_K are different
NUM_EXAMPLES = 1  # TODO or 0?

NUM_THREADS_GENERATE = 3
NUM_THREADS_EVALUATE = 5


MODEL_NAME = 'current-finetuned-model'

START_DATETIME = datetime_str()

# While we have not generated enough samples
# Fetch a random set of authors with at least PAPERS_PER_SAMPLE papers
# Fetch all abstracts of these papers by the author
# Fetch the best matching NUM_EXAMPLES examples in the RAG dataset
# Add a tuple of (author, abstracts, examples) to the samples to generate list

# NUM_THREADS_GENERATE other threads will be running in parallel to generate the samples
# Each thread will fetch one element from the samples to generate list
# Then will call a LLM pipeline on its dedicated GPU to generate the samples
# This call will be with the following parameters:
# - model: 'current-finetuned-model'
# - prompt: to extract with the abstracts and examples
# - temperature: TEMPERATURE
# - top_k: TOP_K_TO_SAMPLE # Generate the top k (different) extracted profiles
# The generated samples will be added to a list of samples to evaluate

# NUM_THREADS_EVALUATE other threads will be running in parallel to evaluate the samples
# Each thread will on startup create a threadlocal database (threadid + timestamp) to store the evaluation samples
# Each thread will fetch one element from the samples to evaluate list
# Then will call a tournament evaluation on the samples with the largest possible LLM
# The evaluation will be written to the threadlocal database with all the preferences


@dataclass(frozen=True)
class SampleToGenerate:
    author: str
    abstracts: list[str]
    examples: list[Example]


@dataclass(frozen=True)
class SampleToEvaluate:
    author: str
    prompt: str
    profiles: list[Profile]


def load_tokenizer(model_name=MODEL_NAME) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name)  # type: ignore


def load_model(model_name=MODEL_NAME, device='cuda') -> AutoPeftModelForCausalLM:
    model = AutoPeftModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model = model.eval()

    return model


samples_to_generate = Queue[SampleToGenerate]()
samples_to_evaluate = Queue[SampleToEvaluate]()

number_preferences_generated = itertools.count()
total_number_preferences_generated = 0


def generate_prompt(abstracts: list[str], examples: list[Example]) -> list[Message]:
    str_abstracts = '\n\n\n'.join(f'Abstract {i + 1}:\n{abstract}' for i, abstract in enumerate(abstracts))

    return [
        SystemMessage(
            content="""You are a helpful research assistant tasked with analyzing scientific abstracts to extract professional competencies. For each abstract, identify the primary domain of expertise and list specific competencies demonstrated by the author. Format your findings as follows:
```
Domain: [Short Domain Description]
Competencies:
- [Competency Name]: [Brief description of how Competency 1 is demonstrated across the abstracts]
- [Competency Name]: [Brief description of how Competency 2 is demonstrated across the abstracts]
...
```
The domain description should be a brief label, summarizing the overall area of expertise. The competencies should be specific skills or knowledge areas demonstrated in the abstracts.
Extract 3 to at most 8 competencies from the abstracts, providing concise descriptions for each.
Your analysis should be neutral, accurate, and solely based on the content of the abstracts provided."""
        ),
        *format_example_messages(examples),
        HumanMessage(
            content=f'Please analyze these scientific abstracts and extract a single professional profile that reflects the competencies and domain of expertise demonstrated throughout. Consider the entire set of abstracts as one cohesive source for a comprehensive competency overview.\n\n{str_abstracts}'
        ),
    ]


def prompt_messages_to_str(tokenizer: PreTrainedTokenizer, messages: list[Message]) -> str:
    return tokenizer.apply_chat_template(conversation=[message.to_dict() for message in messages])  # type: ignore


async def populate_samples_to_generate():
    # While we have not generated enough samples
    # Fetch a random set of authors with at least PAPERS_PER_SAMPLE papers
    # Fetch all abstracts of these papers by the author
    # Fetch the best matching NUM_EXAMPLES examples in the RAG dataset
    # Add a tuple of (author, abstracts, examples) to the samples to generate list

    for query in get_random_english_authors_abstracts(NUM_SAMPLES_TO_GENERATE // TOP_K_TO_SAMPLE, PAPERS_PER_SAMPLE):
        abstracts = '\n\n'.join(query.abstracts)

        examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Example).invoke(abstracts)

        samples_to_generate.put(SampleToGenerate(query.author, query.abstracts, examples))

        while samples_to_generate.qsize() > 50 and total_number_preferences_generated < NUM_SAMPLES_TO_GENERATE:
            await sleep(1)

        if total_number_preferences_generated >= NUM_SAMPLES_TO_GENERATE:
            break


async def process_samples_to_generate(index: int):
    # Each thread will fetch one element from the samples to generate list
    # Then will call a LLM pipeline on its dedicated GPU to generate the samples
    # This call will be with the following parameters:
    # - model: 'current-finetuned-model'
    # - prompt: to extract with the abstracts and examples
    # - temperature: TEMPERATURE
    # - top_k: TOP_K_TO_SAMPLE # Generate the top k (different) extracted profiles
    # The generated samples will be added to a list of samples to evaluate

    tokenizer = load_tokenizer()
    model = load_model(device=f'cuda:{index}')

    while total_number_preferences_generated < NUM_SAMPLES_TO_GENERATE:
        sample_to_generate = samples_to_generate.get()

        prompt_messages = generate_prompt(sample_to_generate.abstracts, sample_to_generate.examples)

        prompt = prompt_messages_to_str(tokenizer, prompt_messages)

        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        outputs = model.generate(
            **inputs,
            num_return_sequences=TOP_K_TO_SAMPLE,
            num_beams=TOP_K_TO_SAMPLE,
            do_sample=True,
            temperature=TEMPERATURE,
            max_new_tokens=650,
        )
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        profiles = [Profile.parse(response) for response in responses]

        samples_to_evaluate.put(SampleToEvaluate(sample_to_generate.author, prompt, profiles))

        while samples_to_evaluate.qsize() > 50:
            await sleep(1)


async def populate_samples_to_evaluate(index: int):
    global total_number_preferences_generated
    # Each thread will on startup create a threadlocal database (threadid + timestamp) to store the evaluation samples
    # Each thread will fetch one element from the samples to evaluate list
    # Then will call a tournament evaluation on the samples with the largest possible LLM
    # The evaluation will be written to the threadlocal database with all the preferences

    tokenizer = load_tokenizer()
    model = load_model(device=f'cuda:{index}')

    db = DPODatabase(f'dpo_{START_DATETIME}_{index}.db')

    while total_number_preferences_generated < NUM_SAMPLES_TO_GENERATE:
        sample_to_evaluate = samples_to_evaluate.get()

        # TODO run tournament evaluation on the samples with the largest possible LLM

        tournament: TournamentNode = None  # type: ignore

        for preference in get_all_preferences(tournament):
            preferred_profile = sample_to_evaluate.profiles[preference.winner]
            other_profile = sample_to_evaluate.profiles[preference.loser]

            db.add_entry(
                sample_to_evaluate.prompt,
                str(preferred_profile),
                str(other_profile),
                EvaluationType.AUTOMATIC,
                sample_to_evaluate.author,
            )

            # once we have generated a preference, we increase the counter and check if we have generated enough preferences
            total_number_preferences_generated = next(number_preferences_generated)


async def main():
    # One thread will be running in parallel to populate the samples to generate
    # NUM_THREADS_GENERATE other threads will be running in parallel to generate the samples
    # NUM_THREADS_EVALUATE other threads will be running in parallel to evaluate the samples

    futures = [populate_samples_to_generate()]

    for i in range(NUM_THREADS_GENERATE):
        futures.append(process_samples_to_generate(i))

    for i in range(NUM_THREADS_EVALUATE):
        futures.append(populate_samples_to_evaluate(i + NUM_THREADS_GENERATE))

    await gather(*futures)

    print(START_DATETIME)


if __name__ == '__main__':
    run(main())
