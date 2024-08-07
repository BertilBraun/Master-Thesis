from math import ceil, log2
from concurrent.futures import Future, ProcessPoolExecutor


from src.log import LogLevel, log
from src.database import get_retriever_getter
from src.papers import get_random_english_authors_abstracts
from src.extraction_custom import prompt_for_extract_from_abstracts_custom
from src.types import Example, Profile
from src.dpo_cluster.defines import *
from src.dpo_cluster.log_gpu_usage import trace_gpu_usage
from src.util import dump_json, json_dumper, log_all_exceptions, text_similarity, timeblock

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


def load_samples_to_generate() -> list[SampleToGenerate]:
    # While we have not generated enough samples
    # Fetch a random set of authors with at least PAPERS_PER_SAMPLE papers
    # Fetch all abstracts of these papers by the author
    # Fetch the best matching NUM_EXAMPLES examples in the RAG dataset
    # Add a tuple of (author, abstracts, examples) to the samples to generate list
    log(f'Populating samples to generate from {START_DATETIME}')

    samples_to_generate: list[SampleToGenerate] = []
    for query in get_random_english_authors_abstracts(NUM_AUTHORS_TO_PROCESS, PAPERS_PER_SAMPLE):
        log(f'Processing query: {query.author}')
        abstracts = '\n\n'.join(query.abstracts)

        examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Example).invoke(abstracts)

        samples_to_generate.append(SampleToGenerate(query.author, query.abstracts, examples))

    return samples_to_generate


def generate_sample(index: int, samples_to_generate: list[SampleToGenerate]) -> list[SampleToEvaluate]:
    tokenizer = get_tokenizer()
    with timeblock(f'Loading model on {index}'):
        model = get_model(
            # BASE_MODEL_ID,  # TODO This should be the current model but for some reason it is not loading (at least not within 30min)
            device=f'cuda:{index}',
            load_in_8bit=True,
        )

    samples_to_evaluate: list[SampleToEvaluate] = []

    for sample in samples_to_generate:
        with log_all_exceptions(f'generate on {index} for {sample.author} failed'):
            with timeblock(f'Generating sample for {sample.author}'):
                samples_to_evaluate.append(process_sample_to_generate_into_sample_to_evaluate(tokenizer, model, sample))

    return samples_to_evaluate


def process_sample_to_generate_into_sample_to_evaluate(
    tokenizer,
    model,
    sample_to_generate: SampleToGenerate,
) -> SampleToEvaluate:
    prompt_messages = prompt_for_extract_from_abstracts_custom(
        sample_to_generate.abstracts, sample_to_generate.examples
    )

    prompt = prompt_messages_to_str(tokenizer, prompt_messages)
    prompt += '\n<|assistant|>\nDomain: "'

    responses = generate(
        tokenizer,
        model,
        prompt,
        num_return_sequences=TOP_K_TO_SAMPLE,
        temperature=TEMPERATURE,
        max_new_tokens=650,
    )

    profiles: list[Profile] = []
    for response in responses:
        with log_all_exceptions(f'Profile parsing failed for response: {response}'):
            profiles.append(Profile.parse('Domain: "' + response))

    # Filter out too similar profiles
    filtered_profiles: list[Profile] = []
    for profile in profiles:
        if not any(
            text_similarity(str(profile), str(filtered_profile)) > 0.95 for filtered_profile in filtered_profiles
        ):
            filtered_profiles.append(profile)

    if len(filtered_profiles) < len(profiles):
        log(f'Filtered out {len(profiles) - len(filtered_profiles)} similar profiles', level=LogLevel.WARNING)

    dump_json(
        {
            'prompt': prompt_messages,
            'profiles': [str(profile) for profile in filtered_profiles],
        },
        f'{OUTPUT_DIR}/generate/{START_DATETIME}/{sample_to_generate.author}.json',
    )

    return SampleToEvaluate(
        author=sample_to_generate.author,
        prompt=prompt,
        abstracts=sample_to_generate.abstracts,
        profiles=filtered_profiles,
    )


if __name__ == '__main__':
    # One thread will be running in parallel to populate the samples to generate
    # NUM_THREADS_GENERATE other threads will be running in parallel to generate the samples

    samples_to_generate = load_samples_to_generate()

    with ProcessPoolExecutor() as executor, json_dumper(get_profile_output_file_path(START_DATETIME)) as dumper:
        trace_future = executor.submit(trace_gpu_usage, f'{OUTPUT_DIR}/gpu_usage/{START_DATETIME}_generate.log')

        NUM_THREADS_GENERATE = cuda.device_count()

        samples_processed = 0
        samples_per_thread = min(ceil(len(samples_to_generate) / NUM_THREADS_GENERATE), 50)

        while samples_processed < len(samples_to_generate):
            eval_futures: list[Future[list[SampleToEvaluate]]] = []
            for i in range(NUM_THREADS_GENERATE):
                this_threads_samples = samples_to_generate[samples_processed : samples_processed + samples_per_thread]
                samples_processed += samples_per_thread

                authors = [sample.author for sample in this_threads_samples]
                log(f'Starting thread {i} to generate {samples_per_thread} samples for authors: {authors}')
                eval_futures.append(executor.submit(generate_sample, i, this_threads_samples))

            for future in eval_futures:
                for preference in future.result():
                    dumper(preference)

        trace_future.cancel()
        executor.shutdown(wait=False)
