from enum import Enum
from typing import Callable, Dict
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel, RunnableLambda


from src.db import DB
from src.papers import get_papers_by_author
from src.util import timeit
from src.types import Profile, Query


Retriever = Runnable[str, Dict[str, str]]
LLM = Runnable[LanguageModelInput, str]


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
    extract: Callable[[Query, Retriever, LLM], Profile]

    def run_for_author(self, author: str, number_of_papers: int = 5) -> ExtractionResult:
        query = get_papers_by_author(author, number_of_papers=number_of_papers)

        # TODO retriever based on self.example_type == ExampleType.POSITIVE

        retriever = RunnableParallel[str](
            # The DB retriever apparently fails if no examples are requested... Therefore the fallback
            examples=DB.as_retriever(self.number_of_examples).with_fallbacks([RunnableLambda(lambda prompt: '')]),
            content=RunnablePassthrough(),
        )

        llm = ChatOpenAI(model=self.model) | StrOutputParser()

        profile = self.extract(query, retriever, llm)

        return ExtractionResult(
            profile=profile,
            titles=query.titles,
            author=query.author,
        )


class ProfileParser(BaseOutputParser[Profile]):
    def parse(self, text: str) -> Profile:
        return Profile.parse(text)


@timeit('Extracting competencies')
def extract_from_abstracts(
    query: Query,
    retriever: Retriever,
    llm: LLM,
) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the abstracts
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptTemplate.from_template("""something with the abstracts {content} and the examples {examples}""")
    extract_chain = retriever | prompt | llm | ProfileParser()

    return extract_chain.invoke('\n\n'.join(query.abstracts))


@timeit('Extracting competencies')
def extract_from_summaries(
    query: Query,
    retriever: Retriever,
    llm: LLM,
) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the summaries
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # Get the summary from the full text
    # TODO examples for summarization?
    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptTemplate.from_template("""something with summarizing the full text {full_text}""")
    summarize_chain = prompt | llm

    summaries = summarize_chain.batch([{'full_text': full_text} for full_text in query.full_texts])

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptTemplate.from_template(
        """something with all the summaries {content} and the examples {examples}"""
    )
    extract_chain = retriever | prompt | llm | ProfileParser()

    return extract_chain.invoke('\n\n'.join(summaries))


@timeit('Extracting competencies')
def extract_from_full_texts(
    query: Query,
    retriever: Retriever,
    llm: LLM,
) -> Profile:
    # We are summarizing one Paper per Prompt, afterwards combining the extracted competences
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptTemplate.from_template("""something with the full text {content} and the examples {examples}""")
    extract_chain = retriever | prompt | llm | ProfileParser()

    profiles = extract_chain.batch([full_text for full_text in query.full_texts])

    # TODO examples for combination?
    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptTemplate.from_template("""something with all the profiles {profiles}""")
    combine_chain = prompt | llm | ProfileParser()

    return combine_chain.invoke({'profiles': profiles})


# --- TODO only fetch the papers once
# --- TODO move to langchain
# TODO how to add restraints to the models? Like stop tokens or max tokens
# --- TODO use langchain retriever for the vector store
# --- TODO function which runs all the instances for a given author
# TODO prompts (test out '---' as a stop token)
# TODO test with ChatPromptTemplate.from_template and ChatPromptTemplate.from_messages
# --- TODO batched
# --- TODO proper full text paper loading
# TODO add the interface to compare the different approaches
# TODO add the automatic comparison of the results based on an LLM
# TODO different indices in the database for extraction examples, summarization examples, and comparison examples
