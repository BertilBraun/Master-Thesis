from dataclasses import dataclass
import json
import os
from typing import Any, Callable

from src.evaluation import get_all_preferences
from src.types import Competency, ExtractedProfile, Profile, RankingResult, TournamentNode


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


def process_tournament(
    tournament: TournamentNode, profiles: dict[int, ExtractedProfile]
) -> dict[EvaluationIdentifier, EvaluationResult]:
    results: dict[EvaluationIdentifier, EvaluationResult] = {}
    for profile in profiles.values():
        results[EvaluationIdentifier.from_profile(profile)] = EvaluationResult()

    for profile in profiles.values():
        results[EvaluationIdentifier.from_profile(profile)] += EvaluationResult(
            total_occurrences=1,
            total_time=profile.extraction_time,
        )

    for preference in get_all_preferences(tournament):
        for i, profile in enumerate(preference.profiles):
            results[EvaluationIdentifier.from_profile(profiles[profile])] += EvaluationResult(
                num_times_preferred=1 if i == preference.preferred_profile_index else 0,
                total_preference_comparisons=1,
            )

    for node in tournament.all_nodes:
        for i, profile in enumerate(node.match.profiles):
            results[EvaluationIdentifier.from_profile(profiles[profile])] += EvaluationResult(
                num_times_directly_preferred=1 if i == node.match.preferred_profile_index else 0,
                total_direct_preference_comparisons=1,
            )

    return results


def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def parse_tournament_and_profiles_from_json(data: dict) -> tuple[TournamentNode, dict[int, ExtractedProfile]]:
    # convert tournament dictionary to TournamentNode
    def process_node(node_data: dict) -> TournamentNode:
        node = TournamentNode(
            match=RankingResult(
                profiles=node_data['match']['profiles'],
                preferred_profile_index=node_data['match']['preferred_profile_index'],
                reasoning=node_data['match'].get('reasoning', None),
            )
        )
        for child_data in node_data['children']:
            node.children.append(process_node(child_data))
        return node

    tournament = process_node(data['tournament'])

    # convert profiles dictionary to dict[int, ExtractedProfile]
    profiles = {}
    for key, profile_data in data['profiles'].items():
        profiles[int(key)] = ExtractedProfile(
            profile=Profile(
                domain=profile_data['profile']['domain'],
                competencies=[
                    Competency(competency['name'], competency['description'])
                    for competency in profile_data['profile']['competencies']
                ],
            ),
            model=profile_data['model'],
            number_of_examples=profile_data['number_of_examples'],
            extraction_function=profile_data['extraction_function'],
            extraction_time=profile_data['extraction_time'],
        )

    return tournament, profiles


def get_all_json_files(directory: str) -> list[str]:
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')]


def get_all_jsons():
    for file_path in get_all_json_files('results'):
        yield load_json(file_path)

    # TODO: Re-enable this
    # import src.defines
    # from src.dpo.jsonbin import JsonBin
    # jsonbin = JsonBin(src.defines.JSONBIN_API_KEY)
    # for bin_id in jsonbin.bins():
    #     yield jsonbin.bin(bin_id)


if __name__ == '__main__':
    # load the evaluation results
    # get all the preferences from the tournaments

    # Extract:
    # - Average extraction time per model
    # - For each Model - how often was it preferred
    # - For each extraction method - how often was it preferred
    # - For number of examples - how often was it preferred
    # - For each extraction method and model - how long did extraction take

    results: dict[EvaluationIdentifier, EvaluationResult] = {}

    for data in get_all_jsons():
        tournament, profiles = parse_tournament_and_profiles_from_json(data)
        for evaluation_identifier, evaluation_result in process_tournament(tournament, profiles).items():
            if evaluation_identifier in results:
                results[evaluation_identifier] += evaluation_result
            else:
                results[evaluation_identifier] = evaluation_result

    for evaluation_identifier, evaluation_result in results.items():
        print(
            f'Statistics for Model: "{evaluation_identifier.model}" - {evaluation_identifier.extraction_method} ({evaluation_identifier.num_examples} examples)'
        )
        print('Average Extraction Time:', evaluation_result.average_time)
        print('Preference Rate:', evaluation_result.preference_rate)
        print('Direct Preference Rate:', evaluation_result.direct_preference_rate)
        print()

    total_number_of_evaluations = sum(result.total_preference_comparisons for result in results.values()) // 2

    def get_stats(getter: Callable[[EvaluationIdentifier], Any]):
        unique_criteria = set(getter(key) for key in results.keys())
        for criterion in sorted(unique_criteria):
            yield criterion, [value for key, value in results.items() if getter(key) == criterion]

    def print_preference_stats(getter: Callable[[EvaluationIdentifier], Any], description: str) -> None:
        for criterion, filtered_results in get_stats(getter):
            total_times_preferred = sum(result.num_times_preferred for result in filtered_results)
            print(
                f'{description}: "{criterion}" - Total Times Preferred: {total_times_preferred} / {total_number_of_evaluations} '
                f'({total_times_preferred / total_number_of_evaluations * 100:.2f}%)'
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
