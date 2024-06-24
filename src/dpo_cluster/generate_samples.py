from math import ceil, log2
from queue import Queue
from asyncio import run, sleep, gather


from src.log import log
from src.database import get_retriever_getter
from src.papers import get_random_english_authors_abstracts
from src.extraction_custom import prompt_for_extract_from_abstracts_custom
from src.types import Example, Profile
from src.dpo_cluster.defines import *
from src.util import dump_json, json_dumper, log_all_exceptions, timeblock

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


START_DATETIME = get_new_datetime_str()


samples_to_generate = Queue[SampleToGenerate]()


def calculate_number_of_authors_to_process() -> int:
    def calculate_P(n: int) -> int:
        log2_n = int(log2(n))
        sum_part = sum((2 ** (log2_n - r) * (2 ** (r - 1) - 1)) for r in range(1, log2_n + 1))
        return (n - 1) + sum_part

    return ceil(NUM_SAMPLES_TO_GENERATE / calculate_P(TOP_K_TO_SAMPLE))


NUM_AUTHORS_TO_PROCESS = calculate_number_of_authors_to_process()

done_populating_authors = False


async def populate_samples_to_generate() -> None:
    # While we have not generated enough samples
    # Fetch a random set of authors with at least PAPERS_PER_SAMPLE papers
    # Fetch all abstracts of these papers by the author
    # Fetch the best matching NUM_EXAMPLES examples in the RAG dataset
    # Add a tuple of (author, abstracts, examples) to the samples to generate list
    global done_populating_authors

    for query in get_random_english_authors_abstracts(NUM_AUTHORS_TO_PROCESS, PAPERS_PER_SAMPLE):
        log(f'Processing query: {query.author}')
        abstracts = '\n\n'.join(query.abstracts)

        examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Example).invoke(abstracts)

        samples_to_generate.put(SampleToGenerate(query.author, query.abstracts, examples))

        while samples_to_generate.qsize() > 50:
            await sleep(10)
            log(f'Waiting for samples to generate: {samples_to_generate.qsize()}')

    done_populating_authors = True


async def process_samples_to_generate(index: int) -> None:
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

    with json_dumper(get_profile_output_file_path(START_DATETIME, index)) as dumper:
        while not done_populating_authors or not samples_to_generate.empty():
            with log_all_exceptions('generate'):
                sample_to_generate = samples_to_generate.get(timeout=10)
                log(f'Generating sample for {sample_to_generate.author}')

                with timeblock(f'Generating sample for {sample_to_generate.author}'):
                    sample = process_sample_to_generate_into_sample_to_evaluate(
                        tokenizer,
                        model,
                        sample_to_generate,
                    )

                dumper(sample)


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
        author=sample_to_generate.author,
        prompt=prompt,
        abstracts=sample_to_generate.abstracts,
        profiles=profiles,
    )


async def main():
    # One thread will be running in parallel to populate the samples to generate
    # NUM_THREADS_GENERATE other threads will be running in parallel to generate the samples

    await gather(
        populate_samples_to_generate(),
        *[process_samples_to_generate(i) for i in range(NUM_THREADS_GENERATE)],
    )


if __name__ == '__main__':
    run(main())
