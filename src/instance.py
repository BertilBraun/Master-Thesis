from tqdm import tqdm
from typing import Callable, Protocol
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable


from src.papers import get_papers_by_author
from src.db import DB
from src.util import timeit
from src.log import LogLevel, log
from src.types import Profile, Example, Query


# Returns a list of Examples based on the content parameter
class ExampleGetter(Protocol):
    def __call__(self, content: str) -> list[Example]:
        ...


LLM = Runnable[LanguageModelInput, str]


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
    # - Good vs Bad Examples (best matches in VectorDB and worst matches in DB)

    model: str  # Identifier from Hugging Face or "OpenAI/" + Model Name
    number_of_examples: int  # 0, 1, 2, 3, 4, 5
    good_or_bad_examples: bool  # Good (True) or Bad (False) Examples
    extract: Callable[[Query, ExampleGetter, LLM], Profile]

    def _get_example_getter(self) -> ExampleGetter:
        if self.good_or_bad_examples:
            return lambda content: DB.search(content, limit=self.number_of_examples)
        else:
            return lambda content: DB.search_negative(content, limit=self.number_of_examples)

    def run_for_author(self, author: str, number_of_papers: int = 5) -> ExtractionResult:
        query = get_papers_by_author(author, number_of_papers=number_of_papers)

        profile = self.extract(
            query,
            self._get_example_getter(),
            ChatOpenAI(model=self.model) | StrOutputParser(),
        )

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
    example_getter: ExampleGetter,
    llm: LLM,
) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the abstracts
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    examples = example_getter('\n\n'.join(query.abstracts))

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptTemplate.from_template(
        """something with the abstracts {abstracts} and the examples {examples}"""
    )

    return (prompt | llm | ProfileParser()).invoke({'abstracts': query.abstracts, 'examples': examples})


@timeit('Extracting competencies')
def extract_from_summaries(
    query: Query,
    example_getter: ExampleGetter,
    llm: LLM,
) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the summaries
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # Get the summary from the full text
    # TODO examples?
    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptTemplate.from_template("""something with summarizing the full text {full_text}""")
    summaries = (prompt | llm).batch(
        [
            {
                'full_text': full_text,
            }
            for full_text in query.full_texts
        ]
    )

    examples = example_getter('\n\n'.join(summaries))

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptTemplate.from_template(
        """something with all the summaries {summaries} and the examples {examples}"""
    )

    return (prompt | llm | ProfileParser()).invoke({'summaries': summaries, 'examples': examples})


@timeit('Extracting competencies')
def extract_from_full_texts(
    query: Query,
    example_getter: ExampleGetter,
    llm: LLM,
) -> Profile:
    # We are summarizing one Paper per Prompt, afterwards combining the extracted competences
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptTemplate.from_template(
        """something with the full text {full_text} and the examples {examples}"""
    )
    all_examples = [example_getter(full_text) for full_text in query.full_texts]

    profiles = (prompt | llm | ProfileParser()).batch(
        [
            {
                'full_text': full_text,
                'examples': examples,
            }
            for full_text, examples in zip(query.full_texts, all_examples)
        ]
    )

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = ChatPromptTemplate.from_template("""something with all the profiles {profiles}""")

    return (prompt | llm | ProfileParser()).invoke({'profiles': profiles})


# --- TODO function which runs all the instances for a given author
# TODO prompts (test out '---' as a stop token)
# TODO batched
# --- TODO proper full text paper loading
# TODO add the interface to compare the different approaches
# TODO add the automatic comparison of the results based on an LLM
# TODO different indices in the database for extraction examples, summarization examples, and comparison examples
