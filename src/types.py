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
        # Return the text between the first occurrence of 'Domain:' and the next '\n'
        assert 'Domain:' in text, f'Domain not found in text: {text}'
        return text.split('Domain:')[1].split('\n')[0].strip().strip('"').strip()

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
class Evaluation:
    paper_text: str
    profile: Profile
    reasoning: str  # let the model generate the reasoning before returning the score
    score: int

    def __str__(self) -> str:
        return f"""Paper Text: {self.paper_text}\n\n\nProfile: {self.profile}\n\n\nReasoning: {self.reasoning}\n\nScore: {self.score}"""

    @staticmethod
    def parse_reasoning(text: str) -> str:
        # Return the text between the first occurrence of 'Reasoning: ' and the next '\n\nScore:'
        if 'Reasoning:' not in text:
            log(f'Invalid reasoning format: {text}.', level=LogLevel.WARNING)
            return ''

        return text.split('Reasoning:')[1].split('\n\nScore:')[0]

    @staticmethod
    def parse_evaluation_score(text: str) -> int:
        # Return the number after the first occurrence of 'Score: '
        match = re.search(r'Score: (\d+)', text)
        if match:
            number = int(match.group(1))
            if 0 <= number <= 100:
                return number

        log(f'Invalid score format: {text}. Trying to find a number...', level=LogLevel.WARNING)

        # Return the last occurrence of a number between 0 and 100
        match = re.findall(r'\d+', text)
        if match:
            for number in match[::-1]:
                number = int(number)
                if 0 <= number <= 100:
                    return number

        assert False, f'Invalid score format: {text}'

    @staticmethod
    def parse(text: str) -> Evaluation:
        # Return the text between the first occurrence of 'Paper Text: ' and the next '\n\n\nProfile:'
        paper_text, rest_text = text.split('Paper Text: ')[1].split('\n\n\nProfile:', maxsplit=1)

        return Evaluation(
            paper_text=paper_text,
            profile=Profile.parse(rest_text),
            reasoning=Evaluation.parse_reasoning(rest_text),
            score=Evaluation.parse_evaluation_score(rest_text),
        )


@dataclass(frozen=True)
class Combination:
    input_profiles: list[Profile]
    combined_profile: Profile

    def __str__(self) -> str:
        input_profiles = '\n\n'.join(str(profile) for profile in self.input_profiles)
        return f"""Input Profiles:\n{input_profiles}\n\nCombined Profile:\n{self.combined_profile}"""

    @staticmethod
    def parse(text: str) -> Combination:
        # Return the text between the first occurrence of 'Input Profiles:\n' and the next '\n\nCombined Profile:'
        input_profiles = text.split('Input Profiles:\n')[1].split('\n\nCombined Profile:')[0]

        # Return the text between the first occurrence of '\n\nCombined Profile:\n' and the end of the text
        combined_profile = text.split('\n\nnCombined Profile:\n')[1]

        return Combination(
            input_profiles=[Profile.parse(profile) for profile in input_profiles.split('\n\n')],
            combined_profile=Profile.parse(combined_profile),
        )


@dataclass(frozen=True)
class Author:
    name: str
    id: str
    count: int

    def __repr__(self) -> str:
        return f"Author(name='{self.name}', id='{self.id}', count={self.count})"


@dataclass(frozen=True)
class Query:
    full_texts: list[str]
    abstracts: list[str]
    titles: list[str]
    author: str

    def __repr__(self) -> str:
        full_texts = ', '.join(f'"""{text}"""' for text in self.full_texts)
        abstracts = ', '.join(f'"""{text}"""' for text in self.abstracts)
        titles = ', '.join(f'"""{text}"""' for text in self.titles)
        return (
            f'Query(full_texts=[{full_texts}], abstracts=[{abstracts}], titles=[{titles}], author="""{self.author}""")'
        )


class ExampleType(Enum):
    POSITIVE = 'positive'  # Good Examples (best matches in VectorDB)
    NEGATIVE = 'negative'  # Bad Examples (worst matches in VectorDB)


@dataclass(frozen=True)
class ExtractionResult:
    profile: Profile
    titles: list[str]
    author: str


DatabaseTypes = TypeVar('DatabaseTypes', Example, Summary, Evaluation, Combination)


class Retriever(Protocol, Generic[DatabaseTypes]):
    def invoke(self, input: str) -> list[DatabaseTypes]:
        ...

    def batch(self, inputs: list[str]) -> list[list[DatabaseTypes]]:
        ...


class RetrieverGetter(Protocol):
    def __call__(self, return_type: Type[DatabaseTypes]) -> Retriever[DatabaseTypes]:
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

    @staticmethod
    def empty_instance() -> Instance:
        return Instance(
            model='',
            number_of_examples=0,
            example_type=ExampleType.POSITIVE,
            extract=lambda q, r, l: Profile(domain='', competencies=[]),  # noqa
        )


@dataclass(frozen=True)
class ExtractedProfile:
    profile: Profile
    instance: Instance


@dataclass(frozen=True)
class AuthorExtractionResult:
    profiles: list[ExtractedProfile]
    titles: list[str]
    author: str
