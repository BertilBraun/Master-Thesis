from collections import Counter, defaultdict
from src.finetuning.logic.finetuning_types import (
    PreferenceSample,
    SampleForFineTuningImprovementEvaluation,
    SampleToEvaluate,
)
from src.util.log import log, ratio
from src.util import create_backup, json_dumper, load_json, log_all_exceptions, timeblock, timeit
from src.logic.types import EvaluationResult, EvaluationResult_from_invalid_response, Ranking, TournamentNode
from src.logic.database import get_retriever_getter
from src.extraction.evaluation import (
    default_round_evaluator,
    get_all_preferences,
    prompt_for_ranking,
    run_tournament_ranking,
)

from src.finetuning.defines import *
from src.finetuning.util.llm_util import parse_llm_from_sysargs


if __name__ == '__main__':
    START_DATETIME = get_previous_datetime_str()

LLM = parse_llm_from_sysargs()


def load_samples_to_evaluate() -> list[SampleToEvaluate]:
    # load samples to generate from the json files into the samples_to_evaluate queue
    file = get_profile_output_file_path(START_DATETIME)
    log(f'Loading samples to evaluate from {file}')
    return load_json(file, SampleToEvaluate)


def evaluate_sample(sample_to_evaluate: SampleToEvaluate) -> list[PreferenceSample]:
    with log_all_exceptions(f'evaluate for {sample_to_evaluate.author} failed'):
        with timeblock(f'Evaluating sample for {sample_to_evaluate.author}'):
            return process_sample_to_evaluate(sample_to_evaluate)
    return []


def process_sample_to_evaluate(sample_to_evaluate: SampleToEvaluate) -> list[PreferenceSample]:
    examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking).invoke(
        '\n\n'.join(sample_to_evaluate.abstracts)
    )

    @timeit(f'author {sample_to_evaluate.author}')
    def match_evaluator(profile1_index: int, profile2_index: int) -> EvaluationResult:
        profile1 = sample_to_evaluate.profiles[profile1_index]
        profile2 = sample_to_evaluate.profiles[profile2_index]

        prompt = prompt_for_ranking(profile1, profile2, examples, sample_to_evaluate.abstracts)
        response = LLM.invoke(prompt, temperature=0.1)

        # The output almost never contains a valid JSON, so we need to parse it manually
        return EvaluationResult_from_invalid_response(response)

    tournament = run_tournament_ranking(
        list(range(len(sample_to_evaluate.profiles))),
        default_round_evaluator(match_evaluator),
        do_shuffle=True,
    )

    return tournament_to_preference_samples(sample_to_evaluate, tournament)


def tournament_to_preference_samples(
    sample_to_evaluate: SampleToEvaluate,
    tournament: TournamentNode,
) -> list[PreferenceSample]:
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


def calculate_aggreement_of_preferences(preference_path: str, other_preference_path: str) -> tuple[int, int]:
    # compare agreement with current preferences
    # map from prompt to (chosen, rejected)
    current_preferences: dict[tuple[str, str, str], int] = defaultdict(int)

    for sample in load_json(preference_path):
        current_preferences[(sample['prompt'], sample['chosen'], sample['rejected'])] += 1

    for sample in load_json(other_preference_path):
        current_preferences[(sample['prompt'], sample['chosen'], sample['rejected'])] -= 1

    counter = Counter(current_preferences.values())

    if counter[2] > 0:
        log(
            f'There are multiple samples ({counter[2]}) with the same prompt, chosen and rejected profiles in the current preferences'
        )
    if counter[-2] > 0:
        log(
            f'There are multiple samples ({counter[-2]}) with the same prompt, chosen and rejected profiles in the other preferences'
        )

    return counter[0] + counter[1], counter[0]


if __name__ == '__main__':
    EVALUATE = True
    if EVALUATE:
        samples_to_evaluate = load_json(
            R'C:\Users\berti\OneDrive\Desktop\Cluster Logs\Samples\all_samples.json', SampleToEvaluate
        )
        log(f'Evaluating {len(samples_to_evaluate)} samples')

        with json_dumper(get_preference_output_file_path(f'TEMPORARY_LOCAL_ONLY_{LLM.model}')) as dumper:
            for sample in samples_to_evaluate:
                for preference in evaluate_sample(sample):
                    dumper(preference)

    exit()
    number_of_samples, number_of_agreements = calculate_aggreement_of_preferences(
        R'C:\Users\berti\OneDrive\Docs\Studium\Semester 8\Masterarbeit\Master-Thesis\dpo_output\preferences\TEMPORARY_LOCAL_ONLY_dev-llama-3-large.json',
        R'C:\Users\berti\OneDrive\Docs\Studium\Semester 8\Masterarbeit\Master-Thesis\dpo_output\preferences\TEMPORARY_LOCAL_ONLY_gpt-4o-mini.json',
    )

    log(f'Agreement with current preferences: {ratio(number_of_agreements, number_of_samples)}')


if __name__ == '__main__2':
    # load current preferences and save them as a backup
    preference_path = get_preference_output_file_path(START_DATETIME)
    preferences_existed, preference_backup_path = create_backup(preference_path)
    if not preferences_existed:
        exit(1)

    samples_to_evaluate = load_samples_to_evaluate()
    log(f'Evaluating {len(samples_to_evaluate)} samples')

    with json_dumper(preference_path) as dumper:
        for sample in samples_to_evaluate:
            for preference in evaluate_sample(sample):
                dumper(preference)

    # compare agreement with current preferences
    number_of_samples, number_of_agreements = calculate_aggreement_of_preferences(
        preference_path, preference_backup_path
    )

    log(f'Agreement with current preferences: {ratio(number_of_agreements, number_of_samples)}')

    # then write the samples with the current best model being the baseline model
    # - when the training with the new samples is done, then the current best model (which was trained on the preferences of llama3 70B) will be compared to the new model (trained on the preferences of GPT4o)
    all_samples = load_json(
        SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE,
        SampleForFineTuningImprovementEvaluation,
    )

    # NOTE: load the samples before opening the dumper, otherwise the file will be overwritten already
    with json_dumper(SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE) as dumper:
        for sample in all_samples:
            dumper(sample.with_new_profiles(sample.best_profile_from_last_model, sample.best_profile_from_last_model))
