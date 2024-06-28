import partialjson
from concurrent.futures import Future, ProcessPoolExecutor

from src.log import log
from src.database import get_retriever_getter
from src.evaluation import get_all_preferences, prompt_for_ranking, run_tournament_ranking
from src.types import EvaluationResult, EvaluationResult_from_invalid_response, Ranking
from src.dpo_cluster.defines import *
from src.dpo_cluster.log_gpu_usage import trace_gpu_usage
from src.util import dump_json, json_dumper, load_json, log_all_exceptions, timeblock

if __name__ == '__main__':
    START_DATETIME = get_previous_datetime_str()


def load_samples_to_evaluate() -> list[SampleToEvaluate]:
    # load samples to generate from the json files into the samples_to_evaluate queue
    file = get_profile_output_file_path(START_DATETIME)
    log(f'Loading samples to evaluate from {file}')
    return [SampleToEvaluate.from_json(sample) for sample in load_json(file)]


def evaluate_sample(index: int, samples_to_evaluate: list[SampleToEvaluate]) -> list[PreferenceSample]:
    log(f'Starting evaluation thread {index} for {len(samples_to_evaluate)} samples')
    tokenizer = get_tokenizer(EVALUATION_MODEL_ID)
    model = get_model(
        EVALUATION_MODEL_ID,
        device=f'cuda:{index}',
        load_in_8bit=True,
        use_flash_attention=USE_FLASH_ATTENTION_FOR_EVALUATION,
    )

    preferences: list[PreferenceSample] = []
    for sample in samples_to_evaluate:
        with log_all_exceptions(f'evaluate on {index} for {sample.author} failed'):
            with timeblock(f'Evaluating sample for {sample.author}'):
                preferences += process_sample_to_evaluate(tokenizer, model, sample)

    return preferences


def process_sample_to_evaluate(
    tokenizer,
    model,
    sample_to_evaluate: SampleToEvaluate,
) -> list[PreferenceSample]:
    examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking).invoke(
        '\n\n'.join(sample_to_evaluate.abstracts)
    )

    def round_evaluator(matches: list[tuple[int, int]]) -> list[EvaluationResult]:
        prompts: list[str] = []

        for profile1_index, profile2_index in matches:
            profile1 = sample_to_evaluate.profiles[profile1_index]
            profile2 = sample_to_evaluate.profiles[profile2_index]

            prompt_messages = prompt_for_ranking(profile1, profile2, examples, sample_to_evaluate.abstracts)
            prompt = prompt_messages_to_str(tokenizer, prompt_messages)

            prompts.append(prompt)

        # group by EVALUATION_BATCH_SIZE
        batched_prompts = [
            prompts[i : i + EVALUATION_BATCH_SIZE] for i in range(0, len(prompts), EVALUATION_BATCH_SIZE)
        ]

        # Call batch_generate with the prompts
        responses: list[str] = []
        for batch in batched_prompts:
            responses += batched_generate(
                tokenizer,
                model,
                batch,
                do_sample=False,
                max_new_tokens=350,
            )

        # Log the responses to a file
        dump_json(
            [
                {
                    'prompt': prompt,
                    'response': response,
                }
                for prompt, response in zip(prompts, responses)
            ],
            f'{OUTPUT_DIR}/evaluate/{START_DATETIME}/{sample_to_evaluate.author}_round({len(matches)}).json',
        )

        # Parse the responses and return the results
        evaluation_results: list[EvaluationResult] = []
        for response in responses:
            try:
                evaluation_results.append(partialjson.JSONParser().parse(response))
            except Exception as e:
                log(f'Error parsing response: {response} - {e}')
                evaluation_results.append(EvaluationResult_from_invalid_response(response))

        return evaluation_results

    tournament = run_tournament_ranking(
        list(range(len(sample_to_evaluate.profiles))),
        round_evaluator,
        do_shuffle=True,
    )

    preferences: list[PreferenceSample] = []

    for preference in get_all_preferences(tournament):
        preferred_profile = sample_to_evaluate.profiles[preference.winner]
        other_profile = sample_to_evaluate.profiles[preference.loser]

        preferences.append(
            PreferenceSample(
                prompt=sample_to_evaluate.prompt,
                chosen=str(preferred_profile),
                rejected=str(other_profile),
            )
        )

    return preferences


if __name__ == '__main__':
    # One thread will be running in parallel to populate the samples to evaluate queue
    # NUM_THREADS_EVALUATE other threads will be running in parallel to evaluate the samples

    samples_to_evaluate = load_samples_to_evaluate()

    with ProcessPoolExecutor() as executor, json_dumper(get_preference_output_file_path(START_DATETIME)) as dumper:
        trace_future = executor.submit(trace_gpu_usage, f'{OUTPUT_DIR}/gpu_usage/{START_DATETIME}_evaluate.log')

        while samples_to_evaluate:
            samples_per_thread = min(len(samples_to_evaluate) // NUM_THREADS_EVALUATE, 20)

            eval_futures: list[Future[list[PreferenceSample]]] = []
            for i in range(NUM_THREADS_EVALUATE):
                authors = [sample.author for sample in samples_to_evaluate[:samples_per_thread]]
                log(f'Starting thread {i} to evaluate {samples_per_thread} samples for authors: {authors}')
                eval_futures.append(executor.submit(evaluate_sample, i, samples_to_evaluate[:samples_per_thread]))
                samples_to_evaluate = samples_to_evaluate[samples_per_thread:]

            for future in eval_futures:
                for preference in future.result():
                    dumper(preference)

        trace_future.cancel()
        executor.shutdown(wait=True)
