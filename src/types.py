from __future__ import annotations

import re

from enum import Enum
from typing import Callable, Generic, Protocol, Type, TypeVar
from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessageParam

from src.log import LogLevel, log

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
class Summary:
    full_text: str
    summary: str

    def __str__(self) -> str:
        return f"""Full text:{self.full_text}\n\n\nSummary:{self.summary}"""

    @staticmethod
    def parse(text: str) -> Summary:
        # Return the text between the first occurrence of 'Full text:' and the next '\n\n\nSummary:'
        full_text = text.split('Full text:')[1].split('\n\n\nSummary:')[0]

        # Return the text between the first occurrence of '\n\n\nSummary:' and the end of the text
        summary = text.split('\n\n\nSummary:')[1]

        return Summary(full_text=full_text, summary=summary)


@dataclass(frozen=True)
class EvaluationScore:
    value: int

    def __str__(self) -> str:
        return f"""Score: {self.value}"""

    @staticmethod
    def parse(text: str) -> EvaluationScore:
        # Return the number after the first occurrence of 'Score: '
        assert 'Score: ' in text, f'Score not found in text: {text}'

        match = re.search(r'Score: (\d+)', text)
        if match:
            return EvaluationScore(value=int(match.group(1)))

        log(f'Invalid score format: {text}. Trying to find a number...', level=LogLevel.WARNING)

        # Return the first occurrence of a number
        match = re.search(r'\d+', text)

        assert match, f'Invalid score format: {text}'

        return EvaluationScore(value=int(match.group(0)))


@dataclass(frozen=True)
class Evaluation:
    text: str
    profile: Profile
    score: EvaluationScore

    def __str__(self) -> str:
        return f"""Text: {self.text}\n\n\nProfile: {self.profile}\n\n\nScore: {self.score}"""

    @staticmethod
    def parse(text: str) -> Evaluation:
        # Return the text between the first occurrence of 'Text: ' and the next '\n\n\nProfile:'
        text = text.split('Text: ')[1].split('\n\n\nProfile:')[0]

        # Return the text between the first occurrence of '\n\n\nProfile: ' and the next '\n\n\nScore:'
        profile = text.split('\n\n\nProfile: ')[1].split('\n\n\nScore:')[0]

        return Evaluation(text=text, profile=Profile.parse(profile), score=EvaluationScore.parse(text))


@dataclass(frozen=True)
class Combination:
    input_profiles: list[Profile]
    output_profile: Profile

    def __str__(self) -> str:
        input_profiles = '\n\n'.join(str(profile) for profile in self.input_profiles)
        return f"""Input Profiles:\n{input_profiles}\n\nOutput Profile:\n{self.output_profile}"""

    @staticmethod
    def parse(text: str) -> Combination:
        # Return the text between the first occurrence of 'Input Profiles:\n' and the next '\n\nOutput Profile:'
        input_profiles = text.split('Input Profiles:\n')[1].split('\n\nOutput Profile:')[0]

        # Return the text between the first occurrence of '\n\nOutput Profile:\n' and the end of the text
        output_profile = text.split('\n\nOutput Profile:\n')[1]

        return Combination(
            input_profiles=[Profile.parse(profile) for profile in input_profiles.split('\n\n')],
            output_profile=Profile.parse(output_profile),
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


T = TypeVar('T')


class Retriever(Protocol, Generic[T]):
    def invoke(self, input: str) -> list[T]:
        ...

    def batch(self, inputs: list[str]) -> list[list[T]]:
        ...


class RetrieverGetter(Protocol, Generic[T]):
    def __call__(self, return_type: Type[T]) -> Retriever[T]:
        ...


@dataclass(frozen=True)
class SystemMessage:
    content: str

    def to_dict(self) -> ChatCompletionMessageParam:
        return {'content': self.content, 'role': 'system'}


@dataclass(frozen=True)
class HumanMessage:
    content: str

    def to_dict(self) -> ChatCompletionMessageParam:
        return {'content': self.content, 'role': 'user'}


@dataclass(frozen=True)
class AIMessage:
    content: str

    def to_dict(self) -> ChatCompletionMessageParam:
        return {'content': self.content, 'role': 'assistant'}


@dataclass(frozen=True)
class HumanExampleMessage:
    content: str

    def to_dict(self) -> ChatCompletionMessageParam:
        return {'content': self.content, 'name': 'example_user', 'role': 'system'}


@dataclass(frozen=True)
class AIExampleMessage:
    content: str

    def to_dict(self) -> ChatCompletionMessageParam:
        return {'content': self.content, 'name': 'example_assistant', 'role': 'system'}


Message = SystemMessage | HumanMessage | AIMessage | HumanExampleMessage | AIExampleMessage


class LanguageModel(Protocol):
    def __init__(self, model: str):
        ...

    def invoke(self, prompt: list[Message], /, stop: list[str] | None = None) -> str:
        ...

    def invoke_profile(self, prompt: list[Message]) -> Profile:
        ...

    def batch(self, prompts: list[list[Message]], /, stop: list[str] | None = None) -> list[str]:
        ...


@dataclass(frozen=True)
class Instance:
    # - Different Models (Types and Sizes)
    # - Abstract vs Automatic Summary vs Full Text
    # - Zero- vs One- vs Few-Shot
    # - TODO not yet - Good vs Bad Prompt
    # - TODO not supported by langchain - Good vs Bad Examples (best matches in VectorDB and worst matches in DB)

    model: str  # Identifier from OpenAI/Insomnium
    number_of_examples: int
    example_type: ExampleType
    extract: Callable[[Query, RetrieverGetter, LanguageModel], Profile]


@dataclass(frozen=True)
class ExtractedProfile:
    profile: Profile
    instance: Instance


@dataclass(frozen=True)
class AuthorExtractionResult:
    profiles: list[ExtractedProfile]
    titles: list[str]
    author: str
