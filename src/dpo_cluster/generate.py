import itertools
import partialjson
from torch import cuda
from queue import Queue
from asyncio import run, sleep, gather
from dataclasses import dataclass


from src.log import log
from src.dpo.dpo_database import DPODatabase, EvaluationType
from src.database import get_retriever_getter
from src.papers import get_random_english_authors_abstracts
from src.evaluation import get_all_preferences, prompt_for_ranking, run_tournament_ranking
from src.extraction_custom import prompt_for_extract_from_abstracts_custom
from src.types import EvaluationResult, Example, Profile, Ranking
from src.dpo_cluster.defines import *
from src.util import dump_json, json_dumper, log_all_exceptions

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


START_DATETIME = get_new_datetime_str()


@dataclass(frozen=True)
class SampleToGenerate:
    author: str
    abstracts: list[str]
    examples: list[Example]


@dataclass(frozen=True)
class SampleToEvaluate:
    author: str
    prompt: str
    abstracts: list[str]
    profiles: list[Profile]


samples_to_generate = Queue[SampleToGenerate]()
samples_to_evaluate = Queue[SampleToEvaluate]()

number_preferences_generated = itertools.count()
total_number_preferences_generated = 0


async def populate_samples_to_generate():
    # While we have not generated enough samples
    # Fetch a random set of authors with at least PAPERS_PER_SAMPLE papers
    # Fetch all abstracts of these papers by the author
    # Fetch the best matching NUM_EXAMPLES examples in the RAG dataset
    # Add a tuple of (author, abstracts, examples) to the samples to generate list

    for query in get_random_english_authors_abstracts(NUM_SAMPLES_TO_GENERATE // TOP_K_TO_SAMPLE, PAPERS_PER_SAMPLE):
        log(f'Processing query: {query.author}')
        abstracts = '\n\n'.join(query.abstracts)

        examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Example).invoke(abstracts)

        samples_to_generate.put(SampleToGenerate(query.author, query.abstracts, examples))

        while samples_to_generate.qsize() > 50 and total_number_preferences_generated < NUM_SAMPLES_TO_GENERATE:
            await sleep(10)
            log(f'Waiting for samples to generate: {samples_to_generate.qsize()}')

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

    tokenizer = get_tokenizer()
    model = get_model(device=f'cuda:{index}')

    with json_dumper(f'samples_to_generate_{START_DATETIME}_{index}.json') as dumper:
        while total_number_preferences_generated < NUM_SAMPLES_TO_GENERATE:
            with log_all_exceptions('generate'):
                sample_to_generate = samples_to_generate.get()
                log(f'Generating sample for {sample_to_generate.author}')

                sample = process_sample_to_generate_into_sample_to_evaluate(
                    tokenizer,
                    model,
                    sample_to_generate,
                )

                samples_to_evaluate.put(sample)
                dumper(sample)

                while samples_to_evaluate.qsize() > 50:
                    await sleep(1)
                    log(f'Waiting for samples to evaluate: {samples_to_evaluate.qsize()}')


def process_sample_to_generate_into_sample_to_evaluate(
    tokenizer,
    model,
    sample_to_generate: SampleToGenerate,
) -> SampleToEvaluate:
    prompt_messages = prompt_for_extract_from_abstracts_custom(
        sample_to_generate.abstracts, sample_to_generate.examples
    )

    prompt = prompt_messages_to_str(tokenizer, prompt_messages)

    responses = generate(
        tokenizer,
        model,
        prompt,
        num_return_sequences=TOP_K_TO_SAMPLE,
        do_sample=True,
        temperature=TEMPERATURE,
        max_new_tokens=650,
    )

    profiles = [Profile.parse(response) for response in responses]

    dump_json(
        {
            'prompt': prompt_messages,
            'profiles': [str(profile) for profile in profiles],
        },
        f'{OUTPUT_DIR}/sample_to_evaluate_{sample_to_generate.author}_{START_DATETIME}.json',
    )

    return SampleToEvaluate(
        sample_to_generate.author,
        prompt,
        sample_to_generate.abstracts,
        profiles,
    )


async def process_samples_to_evaluate(index: int):
    # Each thread will on startup create a threadlocal database (threadid + timestamp) to store the evaluation samples
    # Each thread will fetch one element from the samples to evaluate list
    # Then will call a tournament evaluation on the samples with the largest possible LLM
    # The evaluation will be written to the threadlocal database with all the preferences

    tokenizer = get_tokenizer(EVALUATION_MODEL_ID)
    model = get_model(EVALUATION_MODEL_ID, load_in_4bit=True)

    db = DPODatabase(f'dpo_{START_DATETIME}_{index}.db')

    while total_number_preferences_generated < NUM_SAMPLES_TO_GENERATE:
        with log_all_exceptions('evaluate'):
            sample_to_evaluate = samples_to_evaluate.get()
            log(f'Evaluating sample for {sample_to_evaluate.author}')

            process_sample_to_evaluate(tokenizer, model, db, sample_to_evaluate)


def process_sample_to_evaluate(
    tokenizer,
    model,
    db: DPODatabase,
    sample_to_evaluate: SampleToEvaluate,
) -> None:
    global total_number_preferences_generated

    examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking).invoke(
        '\n\n'.join(sample_to_evaluate.abstracts)
    )

    def evaluator(profile_index1: int, profile_index2: int) -> EvaluationResult:
        profile1 = sample_to_evaluate.profiles[profile_index1]
        profile2 = sample_to_evaluate.profiles[profile_index2]

        prompt_messages = prompt_for_ranking(profile1, profile2, examples, sample_to_evaluate.abstracts)
        prompt = prompt_messages_to_str(tokenizer, prompt_messages)

        response = generate(
            tokenizer,
            model,
            prompt,
            num_return_sequences=1,
            do_sample=False,
            max_new_tokens=350,
        )[0]

        dump_json(
            {
                'prompt': prompt_messages,
                'response': response,
            },
            f'{OUTPUT_DIR}/evaluation_{profile_index1}_{profile_index2}_{START_DATETIME}.json',
        )

        return partialjson.JSONParser().parse(response)

    tournament = run_tournament_ranking(
        list(range(len(sample_to_evaluate.profiles))),
        evaluator,
        do_shuffle=True,
    )

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

    await populate_samples_to_generate()

    tokenizer = get_tokenizer()
    model = get_model()

    with json_dumper(f'samples_to_generate_{START_DATETIME}.json') as dumper:
        while not samples_to_generate.empty():
            sample_to_generate = samples_to_generate.get()

            sample = process_sample_to_generate_into_sample_to_evaluate(
                tokenizer,
                model,
                sample_to_generate,
            )

            samples_to_evaluate.put(sample)
            dumper(sample)

    del tokenizer
    del model
    cuda.empty_cache()

    tokenizer = get_tokenizer(EVALUATION_MODEL_ID)
    model = get_model(EVALUATION_MODEL_ID, load_in_4bit=True)

    db = DPODatabase(f'dpo_{START_DATETIME}_{0}.db')

    while total_number_preferences_generated < NUM_SAMPLES_TO_GENERATE:
        sample_to_evaluate = samples_to_evaluate.get()

        process_sample_to_evaluate(tokenizer, model, db, sample_to_evaluate)

    log('Finished generating and evaluating samples')
    return

    futures = [populate_samples_to_generate()]

    for i in range(NUM_THREADS_GENERATE):
        futures.append(process_samples_to_generate(i + NUM_THREADS_EVALUATE))

    for i in range(NUM_THREADS_EVALUATE):
        futures.append(process_samples_to_evaluate(i))

    await gather(*futures)


if __name__ == '__main__':
    run(main())
