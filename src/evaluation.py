from typing import Callable
from src.database import get_evaluation_messages, get_ranking_messages_json, get_retriever_getter
from src.types import (
    Evaluation,
    EvaluationResult,
    ExtractedProfile,
    Query,
    Ranking,
    RankingResult,
    SystemMessage,
    HumanMessage,
)
from src.language_model import OpenAILanguageModel


def evaluate_with(
    model: str,
    query: Query,
    extractions: list[ExtractedProfile],
) -> list[EvaluationResult]:
    llm = OpenAILanguageModel(model)

    retriever = get_retriever_getter(max_number_to_retrieve=1)(Evaluation)

    abstracts = '\n\n'.join(query.abstracts)

    prompts = [
        [
            SystemMessage(
                content="""You are a skilled evaluator tasked with assessing the relevance of competency profiles relative to the provided scientific abstracts. Evaluate how well each profile reflects the competencies, themes, and expertise areas mentioned in the abstracts. Structure your response as follows:
```
Evaluation and Reasoning: [Your Evaluation and Reasoning]
Score: [Your Score]
```

The evaluation should:
- Discuss the alignment of the profile's competencies with the themes and expertise areas of the abstracts.
- Highlight specific competencies that are well-represented or lacking in relation to the abstract content.
- Comment on the overall coherence between the profile's domain and the abstracts' focus areas.

The score should:
- Be between 0 and 100, where 100 represents a perfect alignment and 0 indicates no relevance at all.

Your analysis should be detailed, citing specific elements from both the profile and the abstracts to support your evaluation."""
            ),
            # TODO get one high scoring and one low scoring profile as an example?
            *get_evaluation_messages(abstracts, retriever),
            HumanMessage(
                content=f'Please assess the following competency profile in terms of its relevance to these scientific abstracts and provide a relevance score.\n\nAbstracts:\n{abstracts}\n\nProfile Details:\n{extraction.profile}\n\nYour evaluation should include specific examples and reasoning, followed by a score between 0 to 100.'
            ),
        ]
        for extraction in extractions
    ]

    responses = llm.batch(prompts)

    results = [
        EvaluationResult(
            extraction=extraction,
            reasoning=Evaluation.parse_reasoning(response),
            score=Evaluation.parse_evaluation_score(response),
        )
        for extraction, response in zip(extractions, responses)
    ]

    return results


def compare_profiles(
    profile1: ExtractedProfile,
    profile2: ExtractedProfile,
    evaluator: Callable[[ExtractedProfile, ExtractedProfile], str],
) -> tuple[ExtractedProfile, ExtractedProfile, str]:
    # Compare two profiles based using a llm model to determine the winner. Return the winner, loser, and reasoning.

    response = evaluator(profile1, profile2)

    reasoning = Ranking.parse_reasoning_json(response)
    is_profile_1_preferred = Ranking.parse_preferred_profile_json(response)

    if is_profile_1_preferred:
        return profile1, profile2, reasoning
    else:
        return profile2, profile1, reasoning


def get_prompt_for_tournament_ranking(model: str, query: Query) -> Callable[[ExtractedProfile, ExtractedProfile], str]:
    llm = OpenAILanguageModel(model, debug_context_name='tournament_ranking')

    abstracts = '\n\n'.join(query.abstracts)

    retriever = get_retriever_getter(max_number_to_retrieve=1)(Ranking)
    json_examples = get_ranking_messages_json(abstracts, retriever)

    def prompt_for_tournament_ranking(profile1: ExtractedProfile, profile2: ExtractedProfile) -> str:
        prompt = [
            SystemMessage(
                content="""You are a skilled evaluator tasked with assessing the relevance of competency profiles relative to the provided scientific abstracts. Evaluate how well each profile reflects the competencies, themes, and expertise areas mentioned in the abstracts. Then compare the two profiles and determine which one is more relevant to the abstracts. Structure your response as json as follows:
```json
{
    "reasoning": "[Your Evaluation and Reasoning]",
    "preferred_profile": [1 or 2]
}
```"""
            ),
            *json_examples,
            HumanMessage(
                content=f'Please assess the following competency profile in terms of its relevance to these scientific abstracts.\n\nAbstracts:\n{abstracts}\n\n\nProfile 1:\n{profile1.profile}\n\n\nProfile 2:\n{profile2.profile}\n\nYour evaluation must follow this json format:\n'
                + """```json
{
    "reasoning": "[Your Evaluation and Reasoning]",
    "preferred_profile": [1 or 2]
}
```
Be specific and detailed in your reasoning and provide the number of the preferred profile."""
            ),
        ]

        return llm.invoke(prompt, response_format='json_object', stop=['\n\n\n'])

    return prompt_for_tournament_ranking


def tournament_ranking(
    model: str,
    query: Query,
    extractions: list[ExtractedProfile],
) -> tuple[list[RankingResult], list[tuple[ExtractedProfile, int]]]:
    """This function runs a tournament ranking between a list of profiles to determine the rankings of the profiles.
    We run a tournament where profiles are compared in pairs, and the winner moves to the next round.
    The tournament continues until we have a single winner. The ranking is determined by the number of rounds each profile wins.
    """

    assert len(extractions) > 1, 'Tournament ranking requires at least two profiles to compare.'

    evaluator = get_prompt_for_tournament_ranking(model, query)

    current_round = extractions
    ranking_results: list[RankingResult] = []
    # Track depth level of each profile in the tournament
    extraction_levels = {id(extraction): 0 for extraction in extractions}

    round_number = 0
    # Run the tournament until we have one winner
    while len(current_round) > 1:
        next_round: list[ExtractedProfile] = []
        round_number += 1

        # Pair profiles and determine winners for the next round
        for i in range(0, len(current_round) - 1, 2):
            winner, loser, reasoning = compare_profiles(current_round[i], current_round[i + 1], evaluator)
            next_round.append(winner)
            ranking_results.append(RankingResult(preferred_profile=winner, other_profile=loser, reasoning=reasoning))
            extraction_levels[id(winner)] = round_number  # Update winner level

        # If odd number of profiles, last one automatically moves to the next round
        if len(current_round) % 2 == 1:
            next_round.append(current_round[-1])

        current_round = next_round

    # If only one profile, it wins by default
    extraction_levels[id(current_round[0])] = round_number

    # Prepare final ranking list
    ranked_tuples = [(extraction, extraction_levels[id(extraction)]) for extraction in extractions]

    return ranking_results, ranked_tuples
