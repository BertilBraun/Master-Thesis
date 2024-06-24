import partialjson
from queue import Queue
from asyncio import run, gather


from src.log import log
from src.database import get_retriever_getter
from src.evaluation import get_all_preferences, prompt_for_ranking, run_tournament_ranking
from src.types import EvaluationResult, Ranking
from src.dpo_cluster.generate_samples import SampleToEvaluate, get_profile_output_file_path
from src.dpo_cluster.defines import *
from src.util import dump_json, json_dumper, load_json, log_all_exceptions, timeblock

# NUM_THREADS_EVALUATE other threads will be running in parallel to evaluate the samples
# Each thread will on startup create a threadlocal database (threadid + timestamp) to store the evaluation samples
# Each thread will fetch one element from the samples to evaluate list
# Then will call a tournament evaluation on the samples with the largest possible LLM
# The evaluation will be written to the threadlocal database with all the preferences


START_DATETIME = get_previous_datetime_str()


samples_to_evaluate = Queue[SampleToEvaluate]()

done_loading_samples_to_evaluate = False


async def load_samples_to_generate() -> None:
    global done_loading_samples_to_evaluate

    # load samples to generate from the json files into the samples_to_evaluate queue
    for i in range(NUM_THREADS_GENERATE):
        file = get_profile_output_file_path(START_DATETIME, i)
        for sample in load_json(file):
            samples_to_evaluate.put(SampleToEvaluate.from_json(sample))

    done_loading_samples_to_evaluate = True


async def process_samples_to_evaluate(index: int) -> None:
    # Each thread will on startup create a threadlocal database (threadid + timestamp) to store the evaluation samples
    # Each thread will fetch one element from the samples to evaluate list
    # Then will call a tournament evaluation on the samples with the largest possible LLM
    # The evaluation will be written to the threadlocal database with all the preferences

    tokenizer = get_tokenizer(EVALUATION_MODEL_ID)
    model = get_model(
        EVALUATION_MODEL_ID,
        load_in_8bit=True,
        use_flash_attention=USE_FLASH_ATTENTION_FOR_EVALUATION,
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

        try:
            return partialjson.JSONParser().parse(response)
        except Exception as e:
            log(f'Error parsing response: {response}')
            log(e)
            # last number [1|2] is the preferred profile
            last_one = response.rfind('1')
            last_two = response.rfind('2')
            return {'reasoning': response, 'preferred_profile': max(last_one, last_two)}

    tournament = run_tournament_ranking(
        list(range(len(sample_to_evaluate.profiles))),
        evaluator,
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
