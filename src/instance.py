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


# --- TODO only fetch the papers once
# --- TODO move to langchain
# --- TODO use langchain retriever for the vector store
# --- TODO function which runs all the instances for a given author
# --- TODO prompts - Add the task to the system message
# TODO add restraints to the models? Like stop tokens or max tokens. Don't know if it is necessary
# --- TODO Chroma DB overview - what is currently in the database, illegal entries, etc.
# --- TODO different indices in the database for extraction examples, summarization examples, and comparison examples
# --- TODO test with ChatPromptTemplate.from_template and ChatPromptTemplate.from_messages
# --- TODO batched
# --- TODO proper full text paper loading
# TODO fix 500 error on LLM call
# TODO let LLM reason about the quality of the examples before having to score them - also do so in the database model and the examples
# TODO add examples to the database for the different approaches
# --- TODO add the (programmatic) interface to compare the different approaches
# --- TODO add the automatic comparison of the results based on an LLM
# Delayed (just don't eval that) TODO rethink prompts with zero-shot. This can basically not ever work with the current setup where a specific format is expected


# TODO after fixing and adding the above todos, let that run for a few authors and see how the results look like
# TODO after that, see if we can send out the extractions to the authors for review/evaluation
#   Use the expert evaluation as new examples for the automatic evaluation and as new examples for the database
# TODO after that, compare the automatic evaluation with the expert evaluation
#   Can we use the expert evaluation to improve the automatic evaluation?
#   Is the automatic evaluation good enough to approximate the expert evaluation?
# TODO after that, use the automatic evaluation for RLHF training
