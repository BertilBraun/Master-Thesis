from src.database import get_database_as_retriever, get_evaluation_messages
from src.types import Evaluation, EvaluationScore, ExtractedProfile, Query, SystemMessage, HumanMessage
from src.language_model import OpenAILanguageModel


def evaluate_with(model: str, query: Query, profiles: list[ExtractedProfile]) -> list[tuple[ExtractedProfile, int]]:
    llm = OpenAILanguageModel(model)

    retriever = get_database_as_retriever(max_number_to_retrieve=2, return_type=Evaluation)

    abstracts = '\n\n'.join(query.abstracts)

    prompts = [
        [
            SystemMessage(
                content='Evaluate the relevance of the provided competency profiles against the corresponding scientific abstracts. Score each profile based on how well it reflects the competencies, themes, and expertise areas mentioned in the abstracts. Provide a score from 0 to 100, where 100 represents a perfect match and 0 represents no relevance.'
            ),
            *get_evaluation_messages(abstracts, retriever),
            HumanMessage(
                content=f'Please assess the following profile in terms of its relevance to the provided scientific abstracts and give it a relevance score. \n\nAbstracts: {abstracts} \n\nProfile details: {profile.profile}\n\nProvide a score between 0 to 100 based on how well the profile matches the abstracts.'
            ),
        ]
        for profile in profiles
    ]

    responses = llm.batch(prompts)

    scores = [EvaluationScore.parse(response).value for response in responses]

    for profile, score in zip(profiles, scores):
        print('Profile:', profile, 'has score:', score)

    # sort by score
    sorted_scored_profiles = list(sorted(zip(profiles, scores), key=lambda x: x[1], reverse=True))

    # return the sorted profiles in descending order of scores (highest score first)
    return sorted_scored_profiles
