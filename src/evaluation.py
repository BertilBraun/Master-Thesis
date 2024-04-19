from src.types import ExtractedProfile, Query, SystemMessage, HumanMessage
from src.language_model import OpenAILanguageModel


def evaluate_with(model: str, query: Query, profiles: list[ExtractedProfile]):
    llm = OpenAILanguageModel(model)

    prompt = [
        SystemMessage(content='something about comparing the extracted profiles with how well they match the papers'),
        # *get_example_messages_for_one(content, retriever),
        HumanMessage(
            content=f'something about {profiles} and how well they match the {query.abstracts} with some structured output of ranking or scores or something like that'
        ),
    ]

    response = llm.invoke(prompt)

    # TODO parse the response and return the results in a structured way

    return response
