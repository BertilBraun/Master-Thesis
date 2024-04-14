from tqdm import tqdm
from typing import Callable, Protocol
from dataclasses import dataclass

from src.papers import get_papers_by_author
from src.gpt import query_openai, query_transformers
from src.db import DB
from src.util import timeit
from src.log import LogLevel, log
from src.types import Profile, Example, Query


# Returns a list of Examples based on the content parameter
class ExampleGetter(Protocol):
    def __call__(self, content: str) -> list[Example]:
        ...


# Queries the model with the prompt parameter and returns the generated text, additional parameters can be passed as kwargs
class LLMExecution(Protocol):
    def __call__(self, prompt: str, **kwargs) -> str:
        ...


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
    extract: Callable[[Query, ExampleGetter, LLMExecution], Profile]

    def _get_llm_execution(self) -> LLMExecution:
        if self.model.startswith('OpenAI/'):
            return lambda prompt, **kwargs: query_openai(
                prompt,
                model=self.model.removeprefix('OpenAI/'),
                **kwargs,
            )
        else:
            return lambda prompt, **kwargs: query_transformers(
                prompt,
                model=self.model,
                **kwargs,
            )

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
            self._get_llm_execution(),
        )

        return ExtractionResult(
            profile=profile,
            titles=query.titles,
            author=query.author,
        )


@timeit('Extracting competencies')
def extract_from_abstracts(
    query: Query,
    example_getter: ExampleGetter,
    llm_execution: LLMExecution,
) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the abstracts
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    examples = example_getter('\n\n'.join(query.abstracts))

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = f"""something with all the abstracts {query.abstracts} and the examples {examples}"""

    return Profile.parse(llm_execution(prompt))


@timeit('Extracting competencies')
def extract_from_summaries(
    query: Query,
    example_getter: ExampleGetter,
    llm_execution: LLMExecution,
) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the summaries
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    summaries: list[str] = []
    for full_text in tqdm(query.full_texts, desc='Extracting summaries'):
        # Get the summary from the full text
        # TODO examples?
        # TODO prompt with proper formatting based on the models tokenizer
        prompt = f"""something with summarizing the full text {full_text}"""

        # TODO batched?
        summaries.append(llm_execution(prompt, max_new_tokens=1000))

    examples = example_getter('\n\n'.join(summaries))

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = f"""something with all the summaries {summaries} and the examples {examples}"""

    return Profile.parse(llm_execution(prompt))


@timeit('Extracting competencies')
def extract_from_full_texts(
    query: Query,
    example_getter: ExampleGetter,
    llm_execution: LLMExecution,
) -> Profile:
    # We are summarizing one Paper per Prompt, afterwards combining the extracted competences
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    profiles: list[Profile] = []
    for full_text in tqdm(query.full_texts, desc='Extracting profiles'):
        # Get the profile from the full text

        examples = example_getter(full_text)

        # TODO prompt with proper formatting based on the models tokenizer
        prompt = f"""something with the full text {full_text} and the examples {examples}"""

        # TODO batched?
        generated_text = llm_execution(prompt)
        try:
            profiles.append(Profile.parse(generated_text))
        except AssertionError:
            log(f"Couldn't parse the the profile: {generated_text} with the prompt: {prompt}", level=LogLevel.INFO)

    # TODO prompt with proper formatting based on the models tokenizer
    prompt = f"""something with all the profiles {profiles}"""

    return Profile.parse(llm_execution(prompt))


# --- TODO function which runs all the instances for a given author
# TODO prompts (test out '---' as a stop token)
# TODO batched
# --- TODO proper full text paper loading
# TODO add the interface to compare the different approaches
# TODO add the automatic comparison of the results based on an LLM
