from typing import Callable

from src.scripts.expert_evaluation_analysis import get_all_manual_jsons
from src.logic.types import EvaluationResult, RankingResult
from src.logic.database import get_retriever_getter
from src.logic.language_model import OpenAILanguageModel
from src.logic.types.database_types import EvaluationResult_from_invalid_response, Ranking
from src.extraction.evaluation import get_all_preferences, prompt_for_ranking, compare_profiles
from src.finetuning.extract_from_finetuned_model import get_queries_from_evaluation_folder
from src.util.log import ratio


# Function to update Elo ratings based on the results
def get_elo_ratings(results: list[RankingResult], k: float = 32.0) -> dict[int, float]:
    elo_ratings: dict[int, float] = {}

    for result in results:
        profile1_index, profile2_index = result.profiles
        rating1 = elo_ratings.get(profile1_index, 1000.0)
        rating2 = elo_ratings.get(profile2_index, 1000.0)

        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        expected2 = 1 - expected1  # Since expected1 + expected2 = 1

        if result.preferred_profile_index == 1:
            score1, score2 = 1.0, 0.0
        elif result.preferred_profile_index == 2:
            score1, score2 = 0.0, 1.0
        else:  # Draw
            score1 = score2 = 0.5

        # Update ratings
        elo_ratings[profile1_index] = rating1 + k * (score1 - expected1)
        elo_ratings[profile2_index] = rating2 + k * (score2 - expected2)

    return elo_ratings


# Function to run pairwise evaluations
def run_pairwise_evaluations(
    profile_indices: list[int], evaluator: Callable[[int, int], list[EvaluationResult]]
) -> list[RankingResult]:
    results: list[RankingResult] = []

    for i in range(len(profile_indices)):
        for j in range(i + 1, len(profile_indices)):
            profile1_index = profile_indices[i]
            profile2_index = profile_indices[j]
            evaluation = evaluator(profile1_index, profile2_index)
            for result in evaluation:
                results.append(
                    compare_profiles(
                        profile1_index,
                        profile2_index,
                        result,
                    )
                )

    return results


# Main code execution
if __name__ == '__main__':
    BASE_URL = None  # '# TODO: Replace with the base URL of the API'
    API_KEY = '# TODO: Replace with your API key'
    MAX_RETRIES = 1

    NUM_EXAMPLES = 1  # Adjust as needed
    EVALUATE_WITH_CONSISTENCY_CHECK = True  # Set to False to disable consistency check
    EVALUATIONS_FOLDER = 'evaluation/_DONE_DONE'  # Folder containing evaluation queries

    # Initialize LLMs
    LLM1 = OpenAILanguageModel(
        model='model_id_1',  # TODO: Replace with the model ID
        debug_context_name='evaluate_for_elo_and_consistency',
        base_url=BASE_URL,
        api_key=API_KEY,
        max_retries=MAX_RETRIES,
    )
    LLM2 = OpenAILanguageModel(
        model='model_id_2',  # TODO: Replace with the model ID
        debug_context_name='evaluate_for_elo_and_consistency',
        base_url=BASE_URL,
        api_key=API_KEY,
        max_retries=MAX_RETRIES,
    )
    LLM3 = OpenAILanguageModel(
        model='model_id_3',  # TODO: Replace with the model ID
        debug_context_name='evaluate_for_elo_and_consistency',
        base_url=BASE_URL,
        api_key=API_KEY,
        max_retries=MAX_RETRIES,
    )

    llms = [LLM1, LLM2, LLM3]

    all_jsons_from_manual = {data.author: data for data in get_all_manual_jsons()}

    queries, emails = get_queries_from_evaluation_folder(EVALUATIONS_FOLDER)

    total_preferences = 0
    total_matched_preferences = 0

    for query in queries.values():
        sample_abstracts = query.abstracts
        sample_profiles = all_jsons_from_manual[query.author].profiles

        # Retrieve examples (implement this function based on your data)
        examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking).invoke(
            '\n\n'.join(sample_abstracts)
        )

        # Define the match evaluator function
        def match_evaluator(profile1_index: int, profile2_index: int) -> list[EvaluationResult]:
            profile1 = sample_profiles[profile1_index].profile
            profile2 = sample_profiles[profile2_index].profile

            evaluations: list[EvaluationResult] = []
            for i, llm in enumerate(llms):
                prompt = prompt_for_ranking(profile1, profile2, examples, sample_abstracts)
                response = llm.invoke(prompt, temperature=0.1)
                evaluation = EvaluationResult_from_invalid_response(response)
                evaluations.append(evaluation)

            preferred_profiles = [evaluation['preferred_profile'] for evaluation in evaluations]

            if not EVALUATE_WITH_CONSISTENCY_CHECK:
                return evaluations

            if all(p == preferred_profiles[0] for p in preferred_profiles):
                preferred_profile = preferred_profiles[0]
            else:
                preferred_profile = 0  # Draw

            reasoning = '\n\n'.join(f'LLM{i+1} Reasoning:\n{eval["reasoning"]}' for i, eval in enumerate(evaluations))

            return [EvaluationResult(reasoning=reasoning, preferred_profile=preferred_profile)]

        # Run pairwise evaluations
        results = run_pairwise_evaluations(
            profile_indices=list(range(len(sample_profiles))),
            evaluator=match_evaluator,
        )

        # Get Elo ratings based on results
        elo_ratings = get_elo_ratings(results)

        # Sort profiles by Elo rating
        sorted_profiles = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

        # Output the sorted profiles and their ratings
        for profile_index, rating in sorted_profiles:
            print(f'Profile {profile_index + 1}: Elo Rating = {rating}')
            print(
                f'Profile Model: {sample_profiles[profile_index].model} - Profile Method: {sample_profiles[profile_index].extraction_function}\n'
            )

        # Compare all preferences of the manual evaluation: Each preference should occur in the elo ranking in the order of the manual evaluation
        for preference in get_all_preferences(all_jsons_from_manual[query.author].tournament):
            index_of_winner_in_elo = -1
            score_of_winner_in_elo = -1
            index_of_loser_in_elo = -1
            score_of_loser_in_elo = -1
            for i, (profile_index, elo_score) in enumerate(sorted_profiles):
                if profile_index == preference.winner:
                    index_of_winner_in_elo = i
                    score_of_winner_in_elo = elo_score
                if profile_index == preference.loser:
                    index_of_loser_in_elo = i
                    score_of_loser_in_elo = elo_score

            assert (
                index_of_winner_in_elo != -1
                and index_of_loser_in_elo != -1
                and score_of_winner_in_elo != -1
                and score_of_loser_in_elo != -1
            ), 'Winner or loser not found in elo ranking'

            total_preferences += 1
            if index_of_winner_in_elo < index_of_loser_in_elo:
                total_matched_preferences += 1

            if index_of_winner_in_elo > index_of_loser_in_elo:
                print(
                    f'Preference {preference} not satisfied in elo ranking ({index_of_winner_in_elo} > {index_of_loser_in_elo}) (Winner: {score_of_winner_in_elo}, Loser: {score_of_loser_in_elo})'
                )
            else:
                print(
                    f'Preference {preference} satisfied in elo ranking ({index_of_winner_in_elo} < {index_of_loser_in_elo}) (Winner: {score_of_winner_in_elo}, Loser: {score_of_loser_in_elo})'
                )

    print(f'Matched preferences: {ratio(total_matched_preferences, total_preferences)}')
