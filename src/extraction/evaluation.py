import random
from typing import Callable
from src.logic.database import format_ranking_messages, get_retriever_getter
from src.logic.types import (
    EvaluationResult,
    ExtractedProfile,
    Message,
    Profile,
    Query,
    Ranking,
    RankingResult,
    SystemMessage,
    HumanMessage,
    TournamentNode,
)
from src.logic.language_model import OpenAILanguageModel, trim_text_to_token_length


random.seed(1)

# EvaluatorResult is a dictionary with the keys "reasoning" and "preferred_profile"
# "reasoning" is a string with the evaluator's reasoning for the preferred profile
# "preferred_profile" is an integer (1 or 2) indicating which profile is preferred


def compare_profiles(profile1: int, profile2: int, evaluation: EvaluationResult) -> RankingResult:
    # Compare two profiles based using a llm model to determine the winner. Return the winner, loser, and reasoning.
    reasoning = Ranking.parse_reasoning_json(evaluation)
    preferred_profile_index = Ranking.parse_preferred_profile_json(evaluation)

    return RankingResult(
        profiles=(profile1, profile2),
        reasoning=reasoning,
        preferred_profile_index=preferred_profile_index,
    )


def prompt_for_ranking(
    profile1: Profile,
    profile2: Profile,
    examples: list[Ranking],
    abstracts: list[str],
) -> list[Message]:
    str_abstracts = '\n\n\n'.join(f'Abstract {i + 1}:\n{abstract}' for i, abstract in enumerate(abstracts))
    str_abstracts = trim_text_to_token_length(str_abstracts, 5000)

    return [
        SystemMessage(
            content="""You are a skilled evaluator tasked with evaluating the relevance of two competency profiles that were extracted by another system from provided scientific abstracts. Each profile is expected to reflect a specific domain of expertise and list 3 to at most 8 key competencies demonstrated by the author. Your task is to evaluate how well each profile reflects the competencies, themes, and expertise areas mentioned in the abstracts. Compare the two profiles and determine which one is more relevant to the abstracts, structuring your response as a JSON object as follows:
{
    "reasoning": "[Your Evaluation and Reasoning]",
    "preferred_profile": [1 or 2]
}
Your analysis should be neutral, accurate, and detailed, based on the content of the abstracts provided."""
        ),
        *format_ranking_messages(examples),
        HumanMessage(
            content=f"""Please assess the following competency profile in terms of its relevance to these scientific abstracts.

Abstracts:
{str_abstracts}


Profile 1:
{profile1}


Profile 2:
{profile2}


Your evaluation must follow this JSON format:
{{
    "reasoning": "[Your Evaluation and Reasoning]",
    "preferred_profile": [1 or 2]
}}
Be specific and detailed in your reasoning and provide the number of the preferred profile."""
        ),
    ]


def pseudo_tournament_ranking(
    extractions: dict[int, ExtractedProfile],
    do_shuffle: bool = True,
) -> TournamentNode:
    def evaluator(profile1_index: int, profile2_index: int) -> EvaluationResult:
        return EvaluationResult(reasoning='Pseudo evaluation', preferred_profile=1)

    return run_tournament_ranking(list(extractions.keys()), default_round_evaluator(evaluator), do_shuffle=do_shuffle)


def tournament_ranking(
    model: str,
    query: Query,
    extractions: dict[int, ExtractedProfile],
    do_shuffle: bool = True,
) -> TournamentNode:
    llm = OpenAILanguageModel(model, debug_context_name='tournament_ranking')

    examples = get_retriever_getter(max_number_to_retrieve=1)(Ranking).invoke('\n\n'.join(query.abstracts))

    def evaluator(profile1_index: int, profile2_index: int) -> EvaluationResult:
        profile1 = extractions[profile1_index].profile
        profile2 = extractions[profile2_index].profile
        prompt = prompt_for_ranking(profile1, profile2, examples, query.abstracts)

        return llm.invoke(prompt, response_format='json_object', stop=['\n\n\n\n'])  # type: ignore

    return run_tournament_ranking(list(extractions.keys()), default_round_evaluator(evaluator), do_shuffle=do_shuffle)


def default_round_evaluator(
    evaluator: Callable[[int, int], EvaluationResult],
) -> Callable[[list[tuple[int, int]]], list[EvaluationResult]]:
    def eval(matches: list[tuple[int, int]]) -> list[EvaluationResult]:
        return [evaluator(profile1, profile2) for profile1, profile2 in matches]

    return eval


def run_tournament_ranking(
    all_extractions: list[int],
    round_evaluator: Callable[[list[tuple[int, int]]], list[EvaluationResult]],
    do_shuffle: bool = True,
) -> TournamentNode:
    """This function runs a tournament ranking between a list of profiles to determine the rankings of the profiles.
    We run a tournament where profiles are compared in pairs, and the winner moves to the next round.
    The tournament continues until we have a single winner.

    The function returns the root node of the tournament tree.
    """
    # Evaluator is a function that takes two profile indices and returns a dict with the evaluation results { "reasoning": str, "preferred_profile": int}

    assert len(all_extractions) > 1, 'Tournament ranking requires at least two profiles to compare.'

    current_round = list(all_extractions)
    if do_shuffle:
        random.shuffle(current_round)

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
        next_round: list[int] = []
        next_last_round_nodes: list[TournamentNode] = []

        # Pair profiles and determine winners for the next round
        matches = [(current_round[i], current_round[i + 1]) for i in range(0, len(current_round) - 1, 2)]
        evaluations = round_evaluator(matches)
        for (profile1, profile2), evaluation in zip(matches, evaluations):
            ranking_result = compare_profiles(profile1, profile2, evaluation)

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
                    preferred_profile_index=0,
                    reasoning='Only one profile left in the round.',
                )
            )
            add_child(node)
            next_last_round_nodes.append(node)

        current_round = next_round
        last_round_nodes = next_last_round_nodes
        last_round_index = 0

    return last_round_nodes[0]


def get_all_preferences(root: TournamentNode) -> list[RankingResult]:
    preferences: list[RankingResult] = []

    for node in root.all_nodes:
        # The winner profile is preferred over all profiles in the loser bracket
        for loser_profile in node.all_profiles_in_loser_subtree:
            preferences.append(
                RankingResult(
                    profiles=(node.match.winner, loser_profile),
                    reasoning='Automatically preferred over all profiles in the loser bracket.'
                    if loser_profile != node.match.loser
                    else node.match.reasoning,
                    preferred_profile_index=0,
                )
            )

    return preferences
