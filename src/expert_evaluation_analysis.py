from collections import Counter
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.dpo_cluster.extract_from_finetuned_model import get_queries_from_evaluation_folder
from src.log import ratio
from src.evaluation import get_all_preferences
from src.types import AuthorResult, ExtractedProfile
from src.util import load_json


@dataclass(frozen=True)
class EvaluationIdentifier:
    model: str
    extraction_method: str
    num_examples: int

    def __hash__(self) -> int:
        return hash((self.model, self.extraction_method, self.num_examples))

    def __eq__(self, other) -> bool:
        return isinstance(other, EvaluationIdentifier) and hash(self) == hash(other)

    @staticmethod
    def from_profile(profile: ExtractedProfile) -> 'EvaluationIdentifier':
        return EvaluationIdentifier(
            model=profile.model,
            extraction_method=profile.extraction_function,
            num_examples=profile.number_of_examples,
        )


@dataclass(frozen=True)
class EvaluationResult:
    total_occurrences: int = 0
    total_time: float = 0.0
    num_times_preferred: int = 0
    total_preference_comparisons: int = 0
    num_times_directly_preferred: int = 0
    total_direct_preference_comparisons: int = 0

    @property
    def average_time(self):
        return self.total_time / self.total_occurrences

    @property
    def preference_rate(self):
        return self.num_times_preferred / self.total_preference_comparisons

    @property
    def direct_preference_rate(self):
        return self.num_times_directly_preferred / self.total_direct_preference_comparisons

    def __add__(self, other: 'EvaluationResult') -> 'EvaluationResult':
        return EvaluationResult(
            total_occurrences=self.total_occurrences + other.total_occurrences,
            total_time=self.total_time + other.total_time,
            num_times_preferred=self.num_times_preferred + other.num_times_preferred,
            total_preference_comparisons=self.total_preference_comparisons + other.total_preference_comparisons,
            num_times_directly_preferred=self.num_times_directly_preferred + other.num_times_directly_preferred,
            total_direct_preference_comparisons=self.total_direct_preference_comparisons
            + other.total_direct_preference_comparisons,
        )


def process_tournament(author_result: AuthorResult) -> dict[EvaluationIdentifier, EvaluationResult]:
    results: dict[EvaluationIdentifier, EvaluationResult] = {}
    for profile in author_result.profiles.values():
        results[EvaluationIdentifier.from_profile(profile)] = EvaluationResult()

    for profile in author_result.profiles.values():
        results[EvaluationIdentifier.from_profile(profile)] += EvaluationResult(
            total_occurrences=1,
            total_time=profile.extraction_time,
        )

    for preference in get_all_preferences(author_result.tournament):
        for i, profile in enumerate(preference.profiles):
            results[EvaluationIdentifier.from_profile(author_result.profiles[profile])] += EvaluationResult(
                num_times_preferred=1 if i == preference.preferred_profile_index else 0,
                total_preference_comparisons=1,
            )

    for node in author_result.tournament.all_nodes:
        if len(node.match.profiles) < 2 or node.match.profiles[0] == node.match.profiles[1]:
            continue
        for i, profile in enumerate(node.match.profiles):
            results[EvaluationIdentifier.from_profile(author_result.profiles[profile])] += EvaluationResult(
                num_times_directly_preferred=1 if i == node.match.preferred_profile_index else 0,
                total_direct_preference_comparisons=1,
            )

    return results


def get_all_json_files(directory: str) -> list[str]:
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')]


def _get_all_jsons():
    import src.defines
    from src.old_dpo.jsonbin import JsonBin

    jsonbin = JsonBin(src.defines.JSONBIN_API_KEY)
    all_jsons_from_manual = [jsonbin.bin(bin_id) for bin_id in jsonbin.bins()]

    queries, emails = get_queries_from_evaluation_folder('evaluation/_DONE')

    all_jsons_from_automatic = [load_json(f'evaluation/_DONE/{author}/{author}.json') for author in queries.keys()]

    authors = Counter(
        [AuthorResult.from_json(data).author for data in all_jsons_from_manual + all_jsons_from_automatic]
    )

    manuals, automatics = [], []
    for author, count in authors.items():
        if count > 1:
            for data in all_jsons_from_manual:
                if AuthorResult.from_json(data).author == author:
                    manuals.append(data)
            for data in all_jsons_from_automatic:
                if AuthorResult.from_json(data).author == author:
                    automatics.append(data)

    return manuals, automatics


def get_all_manual_jsons():
    manuals, automatics = _get_all_jsons()

    return manuals


def get_all_automatic_jsons():
    manuals, automatics = _get_all_jsons()

    return automatics


def get_evaluation_results(all_jsons: list) -> tuple[dict[EvaluationIdentifier, EvaluationResult], int, int]:
    results: dict[EvaluationIdentifier, EvaluationResult] = {}
    total_times_profile1_preferred = 0
    total_nodes = 0

    for data in all_jsons:
        author_result = AuthorResult.from_json(data)
        for evaluation_identifier, evaluation_result in process_tournament(author_result).items():
            if evaluation_identifier in results:
                results[evaluation_identifier] += evaluation_result
            else:
                results[evaluation_identifier] = evaluation_result

        total_times_profile1_preferred += sum(
            1
            for node in author_result.tournament.all_nodes
            if node.match.preferred_profile_index == 0
            and len(node.match.profiles) == 2
            and node.match.profiles[0] != node.match.profiles[1]
        )
        total_nodes += sum(
            1
            for node in author_result.tournament.all_nodes
            if len(node.match.profiles) == 2 and node.match.profiles[0] != node.match.profiles[1]
        )

    return results, total_times_profile1_preferred, total_nodes


if __name__ == '__main__':
    from src.evaluation import tournament_ranking

    results = [AuthorResult.from_json(data) for data in get_all_automatic_jsons()]
    queries, emails = get_queries_from_evaluation_folder('evaluation/_DONE')
    EVALUATION_MODEL = 'dev-llama-3-large'  # 'alias-large-instruct'  #

    # evaluate self consistency of the leafes
    missmatches = 0
    profile_1_preferred = 0
    times_missmatched_and_profile_1_preferred = 0
    times_missmatched_and_profile_2_preferred = 0
    evaluations = 0
    for result in results:
        query = queries[result.author]
        for node in result.tournament.all_leafes:
            if node.match.profiles[0] == node.match.profiles[1]:
                continue
            print(f'Evaluating {result.author} for {node.match.profiles}')

            res1 = tournament_ranking(
                EVALUATION_MODEL,
                query,
                {
                    node.match.profiles[0]: result.profiles[node.match.profiles[0]],
                    node.match.profiles[1]: result.profiles[node.match.profiles[1]],
                },
                do_shuffle=False,
            )

            res2 = tournament_ranking(
                EVALUATION_MODEL,
                query,
                {
                    node.match.profiles[1]: result.profiles[node.match.profiles[1]],
                    node.match.profiles[0]: result.profiles[node.match.profiles[0]],
                },
                do_shuffle=False,
            )

            print('Res1:', res1.match.winner, 'Res2:', res2.match.winner)
            print('Reasoning1:', res1.match.reasoning)
            print('Reasoning2:', res2.match.reasoning)
            print('\n\n\n\n')

            profile_1_preferred += res1.match.preferred_profile_index == 0
            profile_1_preferred += res2.match.preferred_profile_index == 0
            missmatches += res1.match.winner != res2.match.winner
            if res1.match.winner != res2.match.winner:
                if res1.match.preferred_profile_index == 0:
                    times_missmatched_and_profile_1_preferred += 1
                else:
                    times_missmatched_and_profile_2_preferred += 1
                if res2.match.preferred_profile_index == 0:
                    times_missmatched_and_profile_1_preferred += 1
                else:
                    times_missmatched_and_profile_2_preferred += 1
            evaluations += 1

    print(f'Missmatches: {missmatches} / {evaluations} ({missmatches / evaluations * 100:.2f}%)')
    print(
        f'Profile 1 preferred: {profile_1_preferred} / {evaluations * 2} ({profile_1_preferred / evaluations * 50:.2f}%)'
    )
    print(
        f'Times missmatched and profile 1 preferred: {times_missmatched_and_profile_1_preferred} / {missmatches * 2} ({times_missmatched_and_profile_1_preferred / missmatches * 50:.2f}%)'
    )
    print(
        f'Times missmatched and profile 2 preferred: {times_missmatched_and_profile_2_preferred} / {missmatches * 2} ({times_missmatched_and_profile_2_preferred / missmatches * 50:.2f}%)'
    )


if __name__ == '__main__2':
    # load the evaluation results
    # get all the preferences from the tournaments

    # Extract:
    # - Average extraction time per model
    # - For each Model - how often was it preferred
    # - For each extraction method - how often was it preferred
    # - For number of examples - how often was it preferred
    # - For each extraction method and model - how long did extraction take

    results, total_times_profile1_preferred, total_nodes = get_evaluation_results(get_all_manual_jsons())

    for evaluation_identifier, evaluation_result in results.items():
        print(
            f'Statistics for Model: "{evaluation_identifier.model}" - {evaluation_identifier.extraction_method} ({evaluation_identifier.num_examples} examples)'
        )
        print('Average Extraction Time:', evaluation_result.average_time)
        print('Preference Rate:', evaluation_result.preference_rate)
        print('Direct Preference Rate:', evaluation_result.direct_preference_rate)
        print()

    total_number_of_evaluations = sum(result.total_preference_comparisons for result in results.values()) // 2
    total_number_of_direct_evaluations = (
        sum(result.total_direct_preference_comparisons for result in results.values()) // 2
    )

    def get_stats(getter: Callable[[EvaluationIdentifier], Any], data=results):
        unique_criteria = set(getter(key) for key in data.keys())
        for criterion in sorted(unique_criteria):
            yield criterion, [value for key, value in data.items() if getter(key) == criterion]

    def print_preference_stats(getter: Callable[[EvaluationIdentifier], Any], description: str, data=results) -> None:
        for criterion, filtered_results in get_stats(getter, data=data):
            total_times_preferred = sum(result.num_times_preferred for result in filtered_results)
            total_times_directly_preferred = sum(result.num_times_directly_preferred for result in filtered_results)
            print(
                f'{description}: "{criterion}" - Total Times Preferred: {ratio(total_times_preferred, total_number_of_evaluations)} - Total Times Directly Preferred: {ratio(total_times_directly_preferred, total_number_of_direct_evaluations)}'
            )

    # Print section header
    print('Finer Grained Statistics\n')

    # Extract:
    # - For each Model - how often was it preferred
    # - For each extraction method - how often was it preferred
    # - For number of examples - how often was it preferred
    # - Average extraction time per model
    # - For each extraction method and model - how long did extraction take

    print('Average Extraction Time Per Model:')

    for model, filtered_results in get_stats(lambda x: x.model):
        total_extraction_time = sum(result.total_time for result in filtered_results)
        time_per_extraction = total_extraction_time / sum(result.total_occurrences for result in filtered_results)
        print(f'Model: "{model}" - Time Per Extraction: {time_per_extraction:.2f} seconds')

    print('Average Extraction Time Per Extraction Method:')

    for (model, extraction_method), filtered_results in get_stats(lambda x: (x.model, x.extraction_method)):
        total_extraction_time = sum(result.total_time for result in filtered_results)
        time_per_extraction = total_extraction_time / sum(result.total_occurrences for result in filtered_results)
        print(
            f'Model: "{model}" - Method: "{extraction_method}" - Time Per Extraction: {time_per_extraction:.2f} seconds'
        )

    print('Preference Rate Per Model:')

    # Analyze by model, extraction method, and number of examples using lambdas
    print_preference_stats(lambda x: x.model, 'Model')
    print_preference_stats(lambda x: x.extraction_method, 'Method')
    print_preference_stats(lambda x: x.num_examples, 'Examples')
    print_preference_stats(lambda x: (x.model, x.extraction_method), 'Model, Method')

    print(f'\nTotal Times Profile 1 Preferred: {ratio(total_times_profile1_preferred,total_nodes)}')

    automatic_eval_results, automatic_total_times_profile1_preferred, automatic_total_nodes = get_evaluation_results(
        get_all_automatic_jsons()
    )
    automatic_total_number_of_evaluations = (
        sum(result.total_preference_comparisons for result in automatic_eval_results.values()) // 2
    )
    automatic_total_number_of_direct_evaluations = (
        sum(result.total_direct_preference_comparisons for result in automatic_eval_results.values()) // 2
    )

    def print_correlation_stats(getter: Callable[[EvaluationIdentifier], Any], description: str) -> None:
        mp = {criterion: filtered_results for criterion, filtered_results in get_stats(getter, automatic_eval_results)}

        results = []

        for criterion, filtered_results in get_stats(getter):
            automatic_filtered_results = mp[criterion]
            total_times_preferred = sum(result.num_times_preferred for result in filtered_results)
            automatic_total_times_preferred = sum(result.num_times_preferred for result in automatic_filtered_results)
            total_times_directly_preferred = sum(result.num_times_directly_preferred for result in filtered_results)
            automatic_total_times_directly_preferred = sum(
                result.num_times_directly_preferred for result in automatic_filtered_results
            )

            mean_preferred = np.mean(
                [
                    total_times_preferred / total_number_of_evaluations,
                    automatic_total_times_preferred / automatic_total_number_of_evaluations,
                ]
            )
            mean_directly_preferred = np.mean(
                [
                    total_times_directly_preferred / total_number_of_direct_evaluations,
                    automatic_total_times_directly_preferred / automatic_total_number_of_direct_evaluations,
                ]
            )
            std_preferred = np.std(
                [
                    total_times_preferred / total_number_of_evaluations,
                    automatic_total_times_preferred / automatic_total_number_of_evaluations,
                ]
            )
            std_directly_preferred = np.std(
                [
                    total_times_directly_preferred / total_number_of_direct_evaluations,
                    automatic_total_times_directly_preferred / automatic_total_number_of_evaluations,
                ]
            )

            if (
                total_times_preferred / total_number_of_evaluations
                < automatic_total_times_preferred / automatic_total_number_of_evaluations
            ):
                std_preferred = -std_preferred
            if (
                total_times_directly_preferred / total_number_of_direct_evaluations
                < automatic_total_times_directly_preferred / automatic_total_number_of_direct_evaluations
            ):
                std_directly_preferred = -std_directly_preferred

            results.append(
                {
                    description: criterion,
                    'Mean': f'{mean_preferred:.2f}',
                    'STD': f'{std_preferred:.2f}',
                    'DMean': f'{mean_directly_preferred:.2f}',
                    'DSTD': f'{std_directly_preferred:.2f}',
                    'TTP': f'{ratio(total_times_preferred, total_number_of_evaluations)}',
                    'TTDP': f'{ratio(total_times_directly_preferred, total_number_of_direct_evaluations)}',
                }
            )

        df_results = pd.DataFrame(results)
        df_results.set_index([description], inplace=True)
        print(df_results)

    print_correlation_stats(lambda x: x.model, 'Model')
    print_correlation_stats(lambda x: x.extraction_method, 'Method')
    print_correlation_stats(lambda x: (x.model, x.extraction_method), 'Model, Method')

    print(
        '\n\nSTD is positive if the manual evaluation is preferred more than the automatic evaluation. F.e. the automatic evaluation is preferred in 60% of the cases and the manual evaluation in 40% of the cases. The STD is then -0.1 with a mean of 0.5.'
    )

    def X(x):
        if x.extraction_method in ('finetuning', 'extract_from_abstracts_custom'):
            return x.model
        return ''

    print_preference_stats(X, 'Model, Method')
    print_preference_stats(X, 'Model, Method', automatic_eval_results)
