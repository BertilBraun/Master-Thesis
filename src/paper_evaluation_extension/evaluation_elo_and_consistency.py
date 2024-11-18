import time
import random
from typing import Any, Callable, TypedDict
from scipy.stats import spearmanr, kendalltau

import src.defines
from src.logic.jsonbin import JsonBin
from src.logic.types.base_types import Profile
from src.logic.types import EvaluationResult, RankingResult
from src.logic.database import get_retriever_getter
from src.logic.language_model import OpenAILanguageModel
from src.logic.types.database_types import EvaluationResult_from_invalid_response, Ranking
from src.extraction.evaluation import prompt_for_ranking, compare_profiles
from src.util import log


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


class UploadedProfile(TypedDict):
    model_name: str
    profile: dict[str, Any]


class UploadedProfiles(TypedDict):
    author: str
    profiles: list[UploadedProfile]
    abstracts: list[str]


# Main code execution
if __name__ == '__main__':
    # Run with: python -m src.paper_evaluation_extension.evaluation_elo_and_consistency
    BASE_URL = src.defines.LOCAL_AI_ML_PC  # TODO: Replace with the base URL of the API
    API_KEY = src.defines.OPENAI_API_KEY  # TODO: Replace with your API key
    MAX_RETRIES = 1

    # Initialize LLMs
    LLMS = [
        'gemma2-9b-it',
        'llama-3.1-70b-versatile',
        'mixtral-8x7b-32768',
        'llama-3.1-8b-instant',
    ]

    NUM_EXAMPLES = 1  # Adjust as needed
    EVALUATE_WITH_CONSISTENCY_CHECK = True  # Set to False to disable consistency check

    llms = [
        OpenAILanguageModel(
            model=model,
            base_url=src.defines.GROQ_BASE_URL,
            api_key=src.defines.GROQ_API_KEY,
            max_retries=MAX_RETRIES,
            debug_context_name='evaluate_for_elo_and_consistency',
        )
        for model in LLMS
    ]

    json_bin = JsonBin(src.defines.JSONBIN_API_KEY)
    jsons: list[UploadedProfiles] = [json_bin.bin(bin_id) for bin_id in json_bin.bins()]  # type: ignore
    all_jsons_from_manual = {data['author']: data for data in jsons if 'abstracts' in data and 'profiles' in data}

    rohs, taus = [], []

    last_eval_time = time.time() - 60  # Initialize to 60 seconds ago to avoid waiting on the first iteration

    log(f'Running Elo and consistency check for {len(jsons)} queries')

    for author, upload in all_jsons_from_manual.items():
        log(f'Query: {author}')
        sample_abstracts = upload['abstracts']
        sample_profiles = upload['profiles'].copy()  # Make a copy to avoid modifying the original list
        random.shuffle(sample_profiles)

        # Retrieve examples (implement this function based on your data)
        examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking).invoke(
            '\n\n'.join(sample_abstracts)
        )

        log(f'Found {len(examples)} examples')

        # Define the match evaluator function
        def match_evaluator(profile1_index: int, profile2_index: int) -> list[EvaluationResult]:
            global last_eval_time
            # Make sure to wait for at least 1 minute between evaluations
            time_since_last_eval = time.time() - last_eval_time
            if time_since_last_eval < 60:
                print(f'Waiting for {60 - time_since_last_eval} seconds before next evaluation')
                time.sleep(60 - time_since_last_eval)
            last_eval_time = time.time()

            profile1 = Profile.from_json(sample_profiles[profile1_index]['profile'])
            profile2 = Profile.from_json(sample_profiles[profile2_index]['profile'])

            evaluations: list[EvaluationResult] = []
            for llm in llms:
                prompt = prompt_for_ranking(profile1, profile2, examples, sample_abstracts)
                response = llm.invoke(prompt, temperature=0.1)
                evaluations.append(EvaluationResult_from_invalid_response(response))

            # TODO also evaluate wheather the model agrees with the flipped profile orders

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
        log(f'Running pairwise evaluations for {len(sample_profiles)} profiles')
        results = run_pairwise_evaluations(
            profile_indices=list(range(len(sample_profiles))),
            evaluator=match_evaluator,
        )
        log('Pairwise evaluations complete')

        # Get Elo ratings based on results
        elo_ratings = get_elo_ratings(results)

        # Sort profiles by Elo rating
        sorted_profiles = list(sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True))

        # Output the sorted profiles and their ratings
        for profile_index, rating in sorted_profiles:
            print(f'Profile {profile_index + 1}: Elo Rating = {rating}')
            print(f'Profile Model: {sample_profiles[profile_index]["model_name"]}')

        # compare the sorting order of the elo ratings with the expert evaluation from jsonbin
        X = [profile_index for profile_index, _ in sorted_profiles]
        Y = [sample_profiles.index(profile) for profile in upload['profiles']]
        rho, p_value = spearmanr(X, Y)
        tau, p_value = kendalltau(X, Y)
        print(f'Spearman correlation: {rho} (p-value: {p_value})')
        print(f'Kendall tau: {tau} (p-value: {p_value})')
        rohs.append(rho)
        taus.append(tau)

    print(f'Mean Spearman correlation: {sum(rohs) / len(rohs)}')
    print(f'Mean Kendall tau: {sum(taus) / len(taus)}')
