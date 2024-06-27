from time import sleep
from math import ceil, log2
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Generator


from src.log import log
from src.database import get_retriever_getter
from src.papers import get_random_english_authors_abstracts
from src.extraction_custom import prompt_for_extract_from_abstracts_custom
from src.types import Example, Profile
from src.dpo_cluster.defines import *
from src.dpo_cluster.log_gpu_usage import trace_gpu_usage
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


if __name__ == '__main__':
    START_DATETIME = get_new_datetime_str()


def calculate_number_of_authors_to_process() -> int:
    def calculate_P(n: int) -> int:
        log2_n = int(log2(n))
        sum_part = sum((2 ** (log2_n - r) * (2 ** (r - 1) - 1)) for r in range(1, log2_n + 1))
        return (n - 1) + sum_part

    return ceil(NUM_SAMPLES_TO_GENERATE / calculate_P(TOP_K_TO_SAMPLE))


NUM_AUTHORS_TO_PROCESS = calculate_number_of_authors_to_process()


def load_samples_to_generate() -> Generator[SampleToGenerate, None, None]:
    # While we have not generated enough samples
    # Fetch a random set of authors with at least PAPERS_PER_SAMPLE papers
    # Fetch all abstracts of these papers by the author
    # Fetch the best matching NUM_EXAMPLES examples in the RAG dataset
    # Add a tuple of (author, abstracts, examples) to the samples to generate list
    log(f'Populating samples to generate from {START_DATETIME}')

    for query in get_random_english_authors_abstracts(NUM_AUTHORS_TO_PROCESS, PAPERS_PER_SAMPLE):
        log(f'Processing query: {query.author}')
        abstracts = '\n\n'.join(query.abstracts)

        examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Example).invoke(abstracts)

        yield SampleToGenerate(query.author, query.abstracts, examples)


def generate_sample(
    tokenizer,
    model,
    sample_to_generate: SampleToGenerate,
) -> SampleToEvaluate:
    with log_all_exceptions('generate'):
        log(f'Generating sample for {sample_to_generate.author}')

        with timeblock(f'Generating sample for {sample_to_generate.author}'):
            return process_sample_to_generate_into_sample_to_evaluate(tokenizer, model, sample_to_generate)


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
        f'{OUTPUT_DIR}/generate/{START_DATETIME}/{sample_to_generate.author}.json',
    )

    return SampleToEvaluate(
        author=sample_to_generate.author,
        prompt=prompt,
        abstracts=sample_to_generate.abstracts,
        profiles=profiles,
    )


if __name__ == '__main__':
    # One thread will be running in parallel to populate the samples to generate
    # NUM_THREADS_GENERATE other threads will be running in parallel to generate the samples

    tokenizer = get_tokenizer()
    models = [get_model(device=f'cuda:{i}') for i in range(NUM_THREADS_GENERATE)]

    eval_futures: list[list[Future[SampleToEvaluate]]] = [[] for _ in range(NUM_THREADS_GENERATE)]

    with ProcessPoolExecutor(max_workers=NUM_THREADS_GENERATE + 1) as executor:
        trace_future = executor.submit(trace_gpu_usage, f'{OUTPUT_DIR}/gpu_usage/{START_DATETIME}_generate.log')

        with json_dumper(get_profile_output_file_path(START_DATETIME)) as dumper:
            for sample in load_samples_to_generate():
                # find the thread with the least number of samples in the queue
                min_index = min(range(NUM_THREADS_GENERATE), key=lambda i: len(eval_futures[i]))
                log(f'Submitting sample to generate for {sample.author} to thread {min_index}')
                eval_futures[min_index].append(executor.submit(generate_sample, tokenizer, models[min_index], sample))

                # load balancing - keep 10 samples per thread in a queue (submitted to the executor) before loading more
                while any(len(futures) >= 10 for futures in eval_futures):
                    sleep(1)
                    for i, futures in enumerate(eval_futures):
                        # write the results to the file, then remove the futures
                        for future in futures:
                            if future.done():
                                dumper(future.result())

                        eval_futures[i] = [future for future in futures if not future.done()]

        trace_future.cancel()
        executor.shutdown(wait=True)
