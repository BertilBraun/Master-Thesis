from enum import Enum
from typing import Callable
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.db import DB
from src.papers import get_papers_by_author
from src.util import timeit, to_flat_list
from src.types import Profile, Query, Example


Retriever = Runnable[str, list[Example]]
LanguageModel = Runnable[LanguageModelInput, str]


class ExampleType(Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'


@dataclass(frozen=True)
class ExtractionResult:
    profile: Profile
    titles: list[str]
    author: str


@dataclass(frozen=True)
class Instance:
    # - Different Models (Types and Sizes)
    # - Abstract vs Automatic Summary vs Full Text
    # - Zero- vs One- vs Few-Shot
    # - TODO not yet - Good vs Bad Prompt
    # - TODO not supported by langchain - Good vs Bad Examples (best matches in VectorDB and worst matches in DB)

    model: str  # Identifier from OpenAI or Insomnium
    number_of_examples: int  # 0, 1, 2, 3, 4, 5
    example_type: ExampleType  # Good ('positive') or Bad ('negative') Examples
    extract: Callable[[Query, Retriever, LanguageModel], Profile]

    def run_for_author(self, author: str, number_of_papers: int = 5) -> ExtractionResult:
        query = get_papers_by_author(author, number_of_papers=number_of_papers)

        # TODO retriever based on self.example_type == ExampleType.POSITIVE

        retriever = DB.as_retriever(self.number_of_examples).with_fallbacks([RunnableLambda(lambda query: [])])

        llm = ChatOpenAI(model=self.model) | StrOutputParser()

        profile = self.extract(query, retriever, llm)

        return ExtractionResult(
            profile=profile,
            titles=query.titles,
            author=query.author,
        )


def get_example_messages_for_one(content: str, retriever: Retriever) -> list[HumanMessage | AIMessage]:
    return get_example_messages([content], retriever)[0]


def get_example_messages(contents: list[str], retriever: Retriever) -> list[list[HumanMessage | AIMessage]]:
    all_examples = retriever.batch(contents)
    return [
        to_flat_list(
            [[HumanMessage(content=example.abstract), AIMessage(content=str(example.profile))] for example in examples]
        )
        for examples in all_examples
    ]


@timeit('Extracting competencies')
def extract_from_abstracts(
    query: Query,
    retriever: Retriever,
    llm: LanguageModel,
) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the abstracts
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    content = '\n\n'.join(query.abstracts)

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptValue(
        messages=[
            SystemMessage(content='something about a professional competency extraction from the abstracts'),
            *get_example_messages_for_one(content, retriever),
            HumanMessage(content=f'something about a professional competency extraction from the abstracts {content}'),
        ]
    )

    return Profile.parse(llm.invoke(prompt))


@timeit('Extracting competencies')
def extract_from_summaries(
    query: Query,
    retriever: Retriever,
    llm: LanguageModel,
) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the summaries
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # Get the summary from the full text
    # TODO examples for summarization?
    # TODO prompt with proper formatting based on the models tokenizer
    prompts: list[LanguageModelInput] = [
        ChatPromptValue(
            messages=[
                SystemMessage(content='something about summarizing the full text'),
                HumanMessage(content=f'something about summarizing the full text {full_text}'),
            ]
        )
        for full_text in query.full_texts
    ]

    summaries = llm.batch(prompts)
    content = '\n\n'.join(summaries)

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptValue(
        messages=[
            SystemMessage(content='something about a professional competency extraction from the summaries'),
            *get_example_messages_for_one(content, retriever),
            HumanMessage(content=f'something about a professional competency extraction from the summaries {content}'),
        ]
    )

    return Profile.parse(llm.invoke(prompt))


@timeit('Extracting competencies')
def extract_from_full_texts(
    query: Query,
    retriever: Retriever,
    llm: LanguageModel,
) -> Profile:
    # We are summarizing one Paper per Prompt, afterwards combining the extracted competences
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    contents = query.full_texts

    # TODO prompt with proper formatting based on the models tokenizer
    prompts: list[LanguageModelInput] = [
        ChatPromptValue(
            messages=[
                SystemMessage(content='something about a professional competency extraction from the full text'),
                *example_messages,
                HumanMessage(
                    content=f'something about a professional competency extraction from the full text {content}'
                ),
            ]
        )
        for content, example_messages in zip(contents, get_example_messages(contents, retriever))
    ]

    llm_profiles = llm.batch(prompts)

    # The parsing and conversion back to string is done to unify the output format
    profiles = [Profile.parse(profile) for profile in llm_profiles]
    profiles_str = '\n\n'.join([str(profile) for profile in profiles])

    # TODO examples for combination?
    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptValue(
        messages=[
            SystemMessage(content='something about good combiner for the profiles'),
            HumanMessage(content=f'something with all the profiles {profiles_str}'),
        ]
    )

    return Profile.parse(llm.invoke(prompt))


# --- TODO only fetch the papers once
# --- TODO move to langchain
# TODO how to add restraints to the models? Like stop tokens or max tokens. ChatModel has a stop parameter at invoke but I don't know how to use it
# --- TODO use langchain retriever for the vector store
# --- TODO function which runs all the instances for a given author
# TODO prompts (test out '---' as a stop token)
# TODO test with ChatPromptTemplate.from_template and ChatPromptTemplate.from_messages
# --- TODO batched
# --- TODO proper full text paper loading
# TODO add the interface to compare the different approaches
# TODO add the automatic comparison of the results based on an LLM
# TODO different indices in the database for extraction examples, summarization examples, and comparison examples
