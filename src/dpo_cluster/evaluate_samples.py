import partialjson
from queue import Queue
from asyncio import run, gather


from src.log import log
from src.database import get_retriever_getter
from src.evaluation import get_all_preferences, prompt_for_ranking, run_tournament_ranking
from src.types import EvaluationResult, Ranking
from src.dpo_cluster.defines import *
from src.util import dump_json, json_dumper, load_json, log_all_exceptions, timeblock

# NUM_THREADS_EVALUATE other threads will be running in parallel to evaluate the samples
# Each thread will on startup create a threadlocal database (threadid + timestamp) to store the evaluation samples
# Each thread will fetch one element from the samples to evaluate list
# Then will call a tournament evaluation on the samples with the largest possible LLM
# The evaluation will be written to the threadlocal database with all the preferences

if __name__ == '__main__':
    START_DATETIME = get_previous_datetime_str()


samples_to_evaluate = Queue[SampleToEvaluate]()

done_loading_samples_to_evaluate = False


async def load_samples_to_generate() -> None:
    global done_loading_samples_to_evaluate
    log(f'Loading samples to evaluate from {START_DATETIME}')

    # load samples to generate from the json files into the samples_to_evaluate queue
    for i in range(NUM_THREADS_GENERATE):
        file = get_profile_output_file_path(START_DATETIME, i)
        log(f'Loading samples to evaluate from {file}')
        for sample in load_json(file):
            samples_to_evaluate.put(SampleToEvaluate.from_json(sample))

    done_loading_samples_to_evaluate = True


async def process_samples_to_evaluate(index: int) -> None:
    # Each thread will on startup create a threadlocal database (threadid + timestamp) to store the evaluation samples
    # Each thread will fetch one element from the samples to evaluate list
    # Then will call a tournament evaluation on the samples with the largest possible LLM
    # The evaluation will be written to the threadlocal database with all the preferences
    log(f'Starting evaluation thread {index}')

    tokenizer = get_tokenizer(EVALUATION_MODEL_ID)
    model = get_model(
        EVALUATION_MODEL_ID,
        load_in_8bit=True,
        use_flash_attention=USE_FLASH_ATTENTION,
    )

    with json_dumper(get_preference_output_file_path(START_DATETIME, index)) as dumper:
        while not done_loading_samples_to_evaluate or not samples_to_evaluate.empty():
            with log_all_exceptions('evaluate'):
                sample_to_evaluate = samples_to_evaluate.get(timeout=10)
                log(f'Evaluating sample for {sample_to_evaluate.author}')

                with timeblock(f'Evaluating sample for {sample_to_evaluate.author}'):
                    preferences = process_sample_to_evaluate(tokenizer, model, sample_to_evaluate)

                for preference in preferences:
                    dumper(preference)


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
            f'{OUTPUT_DIR}/evaluation_{START_DATETIME}_{sample_to_evaluate.author}_round({len(matches)}).json',
        )

        # Parse the responses and return the results
        evaluation_results: list[EvaluationResult] = []
        for response in responses:
            try:
                evaluation_results.append(partialjson.JSONParser().parse(response))
            except Exception as e:
                log(f'Error parsing response: {response} - {e}')
                # last number [1|2] is the preferred profile
                last_one = response.rfind('1')
                last_two = response.rfind('2')
                preferred_profile = 1 if last_two == -1 or last_one > last_two else 2
                evaluation_results.append({'reasoning': response, 'preferred_profile': preferred_profile})

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


async def main():
    # One thread will be running in parallel to populate the samples to evaluate queue
    # NUM_THREADS_EVALUATE other threads will be running in parallel to evaluate the samples

    await gather(
        load_samples_to_generate(),
        *[process_samples_to_evaluate(i) for i in range(NUM_THREADS_EVALUATE)],
    )


if __name__ == '__main__':
    run(main())
