from __future__ import annotations

from enum import Enum
import re

from dataclasses import dataclass
from typing import Callable, Protocol

from langchain_core.runnables import Runnable
from langchain_core.prompt_values import ChatPromptValue

_COMPETENCY_PATTERN = re.compile(r'- (.+?): (.+)')


@dataclass(frozen=True)
class Competency:
    name: str
    description: str

    def __str__(self) -> str:
        return f'{self.name}: {self.description}'

    @staticmethod
    def parse(text: str) -> Competency | None:
        # Return a Competency object if the text matches the pattern '- [COMPETENCY]: [DESCRIPTION]'

        match = _COMPETENCY_PATTERN.match(text)
        if match:
            name, description = match.groups()
            return Competency(name=name, description=description)
        return None


@dataclass(frozen=True)
class Profile:
    domain: str
    competencies: list[Competency]

    def __str__(self) -> str:
        competencies = '\n'.join(f'- {c}' for c in self.competencies)
        return f"""Domain: "{self.domain}"

Competencies:
{competencies}
"""

    @staticmethod
    def _parse_domain(text: str) -> str:
        # Return the text between the first occurrence of 'Domain: "' and the next '"'
        assert 'Domain: "' in text, f'Domain not found in text: {text}'
        return text.split('Domain: "')[1].split('"')[0]

    @staticmethod
    def _parse_competencies(text: str) -> list[Competency]:
        # Returns the list of competencies after the first occurrence of 'Competencies:\n' while the competencies are not empty and the line matches the pattern '- [COMPETENCY]: [DESCRIPTION]'
        assert 'Competencies:\n' in text, f'Competencies not found in text: {text}'

        text = text.split('Competencies:\n')[1]

        competencies: list[Competency] = []
        for line in text.split('\n'):
            competency = Competency.parse(line)
            if competency:
                competencies.append(competency)
            elif competencies:
                # If the line doesn't match the pattern and we have already found competencies, we break
                break

        return competencies

    @staticmethod
    def parse(text: str) -> Profile:
        # NOTE: Throws AssertionError if the text does not contain a valid domain and competencies
        return Profile(
            domain=Profile._parse_domain(text),
            competencies=Profile._parse_competencies(text),
        )


@dataclass(frozen=True)
class Example:
    abstract: str
    profile: Profile

    def __str__(self) -> str:
        return f"""Abstract:
{self.abstract}

{self.profile}
"""

    @staticmethod
    def _parse_abstract(text: str) -> str:
        # Return the text between the first occurrence of 'Abstract:' and the next '\n\n'
        assert 'Abstract:' in text, f'Abstract not found in text: {text}'
        return text.split('Abstract:')[1].split('\n\n')[0]

    @staticmethod
    def parse(text: str) -> Example:
        # NOTE: Throws AssertionError if the text does not contain a valid abstract and profile
        return Example(
            abstract=Example._parse_abstract(text),
            profile=Profile.parse(text),
        )


@dataclass(frozen=True)
class Query:
    full_texts: list[str]
    abstracts: list[str]
    titles: list[str]
    author: str


class ExampleType(Enum):
    POSITIVE = 'positive'  # Good Examples (best matches in VectorDB)
    NEGATIVE = 'negative'  # Bad Examples (worst matches in VectorDB)


@dataclass(frozen=True)
class ExtractionResult:
    profile: Profile
    titles: list[str]
    author: str


Retriever = Runnable[str, list[Example]]


class LanguageModel(Protocol):
    def __init__(self, model: str):
        ...

    def invoke(self, prompt: ChatPromptValue, /, stop: list[str] = []) -> str:
        ...

    def invoke_profile(self, prompt: ChatPromptValue) -> Profile:
        ...

    def batch(self, prompts: list[ChatPromptValue], /, stop: list[str] = []) -> list[str]:
        ...


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


@dataclass(frozen=True)
class ExtractedProfile:
    profile: Profile
    instance: Instance


@dataclass(frozen=True)
class AuthorExtractionResult:
    profiles: list[ExtractedProfile]
    titles: list[str]
    author: str
