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
    TournamentNode,
)
from src.language_model import OpenAILanguageModel
from src.log import LogLevel, log


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
    evaluator: Callable[[ExtractedProfile, ExtractedProfile], dict],
) -> RankingResult:
    # Compare two profiles based using a llm model to determine the winner. Return the winner, loser, and reasoning.

    try:
        response = evaluator(profile1, profile2)
    except Exception as e:
        log(f'Error evaluating profiles: {e}', level=LogLevel.WARNING)
        return RankingResult(profiles=(profile1, profile2), reasoning='Error evaluating profiles', preferred_profile=0)

    reasoning = Ranking.parse_reasoning_json(response)
    is_profile_1_preferred = Ranking.parse_preferred_profile_json(response)

    return RankingResult(
        profiles=(profile1, profile2), reasoning=reasoning, preferred_profile=0 if is_profile_1_preferred else 1
    )


def get_prompt_for_tournament_ranking(model: str, query: Query) -> Callable[[ExtractedProfile, ExtractedProfile], dict]:
    llm = OpenAILanguageModel(model, debug_context_name='tournament_ranking')

    abstracts = '\n\n'.join(query.abstracts)

    retriever = get_retriever_getter(max_number_to_retrieve=1)(Ranking)
    json_examples = get_ranking_messages_json(abstracts, retriever)

    def prompt_for_tournament_ranking(profile1: ExtractedProfile, profile2: ExtractedProfile) -> dict:
        prompt = [
            SystemMessage(
                content="""You are a skilled evaluator tasked with evaluating the relevance of two competency profiles that were extracted by another system from provided scientific abstracts. Each profile is expected to reflect a specific domain of expertise and list 3 to 8 key competencies demonstrated by the author. Your task is to evaluate how well each profile reflects the competencies, themes, and expertise areas mentioned in the abstracts. Compare the two profiles and determine which one is more relevant to the abstracts, structuring your response as follows:
```json
{
    "reasoning": "[Your Evaluation and Reasoning]",
    "preferred_profile": [1 or 2]
}
```
Your analysis should be neutral, accurate, and detailed, based on the content of the abstracts provided."""
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

        return llm.invoke(prompt, response_format='json_object', stop=['\n\n\n\n'])

    return prompt_for_tournament_ranking


def tournament_ranking(
    model: str,
    query: Query,
    extractions: list[ExtractedProfile],
) -> tuple[TournamentNode, list[RankingResult]]:
    """This function runs a tournament ranking between a list of profiles to determine the rankings of the profiles.
    We run a tournament where profiles are compared in pairs, and the winner moves to the next round.
    The tournament continues until we have a single winner.

    The function returns the root node of the tournament tree and a list of all pairwise preferences that were determined during the tournament.
    """

    assert len(extractions) > 1, 'Tournament ranking requires at least two profiles to compare.'

    evaluator = get_prompt_for_tournament_ranking(model, query)

    current_round = extractions

    last_round_index = 0
    last_round_nodes: list[TournamentNode] = []

    def add_child(node: TournamentNode) -> None:
        nonlocal last_round_index
        if last_round_index < len(last_round_nodes):
            child = last_round_nodes[last_round_index]
            last_round_index += 1
            node.children.append(child)

    # Run the tournament until we have one winner
    while len(current_round) > 1:
        next_round: list[ExtractedProfile] = []
        next_last_round_nodes: list[TournamentNode] = []

        # Pair profiles and determine winners for the next round
        for i in range(0, len(current_round) - 1, 2):
            ranking_result = compare_profiles(current_round[i], current_round[i + 1], evaluator)

            node = TournamentNode(match=ranking_result)
            add_child(node)
            add_child(node)
            next_last_round_nodes.append(node)

            next_round.append(ranking_result.winner)

        # If odd number of profiles, last one automatically moves to the next round
        if len(current_round) % 2 == 1:
            next_round.append(current_round[-1])
            node = TournamentNode(
                match=RankingResult(
                    profiles=(current_round[-1], current_round[-1]),
                    preferred_profile=0,
                    reasoning='Only one profile left in the round.',
                )
            )
            add_child(node)
            next_last_round_nodes.append(node)

        current_round = next_round
        last_round_nodes = next_last_round_nodes
        last_round_index = 0

    root = last_round_nodes[0]

    preferences: list[RankingResult] = []

    for node in root.all_nodes:
        preferences.append(node.match)

        # The winner profile is also preferred over all profiles in the loser bracket (unique)
        all_loser_profiles = [
            loser_profile for loser_node in node.all_loser_nodes for loser_profile in loser_node.match.profiles
        ]
        # make sure we don't add the same profile twice (but profiles are not hashable)
        all_loser_profiles_filtered: list[ExtractedProfile] = []
        for loser_profile in all_loser_profiles:
            if loser_profile not in all_loser_profiles_filtered:
                all_loser_profiles_filtered.append(loser_profile)

        for loser_profile in all_loser_profiles_filtered:
            preferences.append(
                RankingResult(
                    profiles=(node.match.winner, loser_profile),
                    reasoning='Automatically preferred over all profiles in the loser bracket.',
                    preferred_profile=0,
                )
            )

    return root, preferences
