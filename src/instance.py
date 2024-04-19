from enum import Enum
from typing import Callable
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.db import DB
from src.log import LogLevel, log
from src.util import timeit
from src.types import Profile, Query, Example
from src.papers import get_papers_by_author


class ExampleType(Enum):
    POSITIVE = 'positive'  # Good Examples (best matches in VectorDB)
    NEGATIVE = 'negative'  # Bad Examples (worst matches in VectorDB)


@dataclass(frozen=True)
class ExtractionResult:
    profile: Profile
    titles: list[str]
    author: str


class LanguageModel:
    def __init__(self, model: str):
        # Note: Cannot chain them because we need to pass the stop tokens to the model
        self.model = ChatOpenAI(model=model)
        self.output_parser = StrOutputParser()

    def invoke(self, prompt: ChatPromptValue, /, stop: list[str] = []) -> str:
        log(f'Running model: {self.model.name}', level=LogLevel.DEBUG)
        log(f'Prompt: {prompt}', level=LogLevel.DEBUG)
        response = self.output_parser.invoke(self.model.invoke(prompt, stop=stop))
        log(f'Response: {response}', level=LogLevel.DEBUG)
        return response

    def invoke_profile(self, prompt: ChatPromptValue) -> Profile:
        # TODO stop tokens
        stop = ['\n\n\n']
        return Profile.parse(self.invoke(prompt, stop=stop))

    def batch(self, prompts: list[ChatPromptValue], /, stop: list[str] = []) -> list[str]:
        log(f'Running batched model: {self.model.name}', level=LogLevel.DEBUG)
        log(f'Prompts: {prompts}', level=LogLevel.DEBUG)
        response = self.output_parser.batch(self.model.batch(prompts, stop=stop))  # type: ignore
        log(f'Response: {response}', level=LogLevel.DEBUG)
        return response


Retriever = Runnable[str, list[Example]]


@dataclass(frozen=True)
class Instance:
    # - Different Models (Types and Sizes)
    # - Abstract vs Automatic Summary vs Full Text
    # - Zero- vs One- vs Few-Shot
    # - TODO not yet - Good vs Bad Prompt
    # - TODO not supported by langchain - Good vs Bad Examples (best matches in VectorDB and worst matches in DB)

    model: str  # Identifier from OpenAI or Insomnium
    number_of_examples: int
    example_type: ExampleType
    extract: Callable[[Query, Retriever, LanguageModel], Profile]

    def run_for_author(self, author: str, number_of_papers: int = 5) -> ExtractionResult:
        query = get_papers_by_author(author, number_of_papers=number_of_papers)

        # TODO retriever based on self.example_type == POSITIVE or NEGATIVE

        retriever = DB.as_retriever(self.number_of_examples).with_fallbacks([RunnableLambda(lambda query: [])])

        llm = LanguageModel(self.model)

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
        [
            message
            for example in examples
            for message in [
                HumanMessage(content=example.abstract),
                AIMessage(content=str(example.profile)),
            ]
        ]
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

    return llm.invoke_profile(prompt)


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
    prompts = [
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

    return llm.invoke_profile(prompt)


@timeit('Extracting competencies')
def extract_from_full_texts(
    query: Query,
    retriever: Retriever,
    llm: LanguageModel,
) -> Profile:
    # We are summarizing one Paper per Prompt, afterwards combining the extracted competences
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # TODO prompt with proper formatting based on the models tokenizer
    prompts = [
        ChatPromptValue(
            messages=[
                SystemMessage(content='something about a professional competency extraction from the full text'),
                *example_messages,
                HumanMessage(
                    content=f'something about a professional competency extraction from the full text {full_text}'
                ),
            ]
        )
        for full_text, example_messages in zip(query.full_texts, get_example_messages(query.full_texts, retriever))
    ]

    llm_profiles = llm.batch(prompts)

    # The parsing and conversion back to string is done to unify the output format
    profiles = '\n\n'.join([str(Profile.parse(profile)) for profile in llm_profiles])

    # TODO examples for combination?
    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptValue(
        messages=[
            SystemMessage(content='something about good combiner for the profiles'),
            HumanMessage(content=f'something with all the profiles {profiles}'),
        ]
    )

    return llm.invoke_profile(prompt)


# --- TODO only fetch the papers once
# --- TODO move to langchain
# --- TODO use langchain retriever for the vector store
# --- TODO function which runs all the instances for a given author
# TODO prompts - Add the task to the system message
# TODO add restraints to the models? Like stop tokens or max tokens. Don't know if it is necessary
# --- TODO test with ChatPromptTemplate.from_template and ChatPromptTemplate.from_messages
# --- TODO batched
# --- TODO proper full text paper loading
# TODO add the interface to compare the different approaches
# TODO add the automatic comparison of the results based on an LLM
# TODO different indices in the database for extraction examples, summarization examples, and comparison examples
