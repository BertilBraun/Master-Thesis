from src.types import (
    Combination,
    Example,
    Profile,
    Query,
    Instance,
    LanguageModel,
    RetrieverGetter,
    Summary,
    SystemMessage,
    HumanMessage,
)
from src.database import (
    get_combination_messages,
    get_example_messages,
    get_retriever_getter,
    get_summary_messages,
)
from src.language_model import OpenAILanguageModel
from src.util import timeit


@timeit('Querying Instance')
def run_query_for_instance(instance: Instance, query: Query) -> Profile:
    # TODO retriever based on instance.example_type == POSITIVE or NEGATIVE

    retriever_getter = get_retriever_getter(instance.number_of_examples)

    llm = OpenAILanguageModel(instance.model)

    return instance.extract(query, retriever_getter, llm)


def extract_from_abstracts(query: Query, retriever: RetrieverGetter, llm: LanguageModel) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the abstracts
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    content = '\n\n'.join(query.abstracts)

    # TODO better prompt
    prompt = [
        SystemMessage(
            content='Extract professional competencies from the following scientific abstracts. Identify key competencies demonstrated within these texts and compile them into a structured profile. Each abstract should be considered as part of a broader set, aiming to provide a comprehensive overview of the competencies across different papers.'
        ),
        *get_example_messages(content, retriever(Example)),
        HumanMessage(
            content=f'Please extract the professional competencies from these scientific abstracts: {content}'
        ),
    ]

    return llm.invoke_profile(prompt)


def extract_from_summaries(query: Query, retriever: RetrieverGetter, llm: LanguageModel) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the summaries
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # Get the summary from the full text
    # TODO better prompt
    prompts = [
        [
            SystemMessage(
                content='Generate a concise summary for the following full text of a scientific paper. The summary should capture the main arguments, methodologies, results, and implications succinctly.'
            ),
            *get_summary_messages(full_text, retriever(Summary)),
            HumanMessage(content=f'Please provide a summary for this complete document text: {full_text}'),
        ]
        for full_text in query.full_texts
    ]

    summaries = llm.batch(prompts)
    content = '\n\n'.join(summaries)

    # TODO better prompt
    prompt = [
        SystemMessage(
            content='Extract professional competencies from the provided summary of the scientific paper. Identify key competencies that are detailed in the summary, and organize them into a structured profile. Each summary should be considered as part of a broader set, aiming to provide a comprehensive overview of the competencies across different papers.'
        ),
        *get_example_messages(content, retriever(Example)),
        HumanMessage(content=f'Please extract the professional competencies from this summary: {content}'),
    ]

    return llm.invoke_profile(prompt)


def extract_from_full_texts(query: Query, retriever: RetrieverGetter, llm: LanguageModel) -> Profile:
    # We are summarizing one Paper per Prompt, afterwards combining the extracted competences
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # TODO better prompt
    prompts = [
        [
            SystemMessage(
                content='Extract professional competencies from the entire text of the provided scientific paper. Identify and list all relevant competencies demonstrated within the text, organizing them into a structured competency profile.'
            ),
            *get_example_messages(full_text, retriever(Example)),
            HumanMessage(
                content=f'Please extract the professional competencies from this complete document text: {full_text}'
            ),
        ]
        for full_text in query.full_texts
    ]

    llm_profiles = llm.batch(prompts)

    # The parsing and conversion back to string is done to unify the output format
    profiles = '\n\n'.join([str(Profile.parse(profile)) for profile in llm_profiles])

    # TODO better prompt
    prompt = [
        SystemMessage(
            content='Combine the following individual competency profiles into a single, comprehensive profile. This unified profile should reflect integrated competencies that encapsulate the essence of all included profiles.'
        ),
        *get_combination_messages(profiles, retriever(Combination)),
        HumanMessage(content=f'Please synthesize these individual profiles into one comprehensive profile: {profiles}'),
    ]

    return llm.invoke_profile(prompt)
