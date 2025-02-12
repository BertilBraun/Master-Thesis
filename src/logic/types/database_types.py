from __future__ import annotations

import re

from typing import Generic, Protocol, Type, TypeVar, TypedDict
from dataclasses import dataclass


from src.logic.types.base_types import Profile
from src.util.log import LogLevel, log


@dataclass(frozen=True)
class Example:
    abstracts: str
    profile: Profile

    def __str__(self) -> str:
        return f"""Abstracts:
{self.abstracts}



{self.profile}
"""

    @staticmethod
    def from_json(data: dict) -> Example:
        return Example(abstracts=data['abstracts'], profile=Profile.from_json(data['profile']))

    @staticmethod
    def _parse_abstracts(text: str) -> str:
        # Return the text between the first occurrence of 'Abstract:' and the next '\n\n'
        assert 'Abstracts:' in text, f'Abstracts not found in text: {text}'
        return text.split('Abstracts:')[1].split('\n\n\n\n')[0]

    @staticmethod
    def parse(text: str) -> Example:
        # NOTE: Throws AssertionError if the text does not contain a valid abstract and profile
        return Example(
            abstracts=Example._parse_abstracts(text),
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


class EvaluationResult(TypedDict):
    reasoning: str
    preferred_profile: int


def EvaluationResult_from_invalid_response(response: str) -> EvaluationResult:
    # last number [0|1|2] is the preferred profile
    if '"preferred_profile":' not in response:
        print(f'Invalid response: {response}')
    last_zero = response.rfind('0')
    last_one = response.rfind('1')
    last_two = response.rfind('2')
    # preferred profile is the one that appears last in the response = argmax(last_zero, last_one, last_two)
    preferred_profile = max(enumerate([last_zero, last_one, last_two]), key=lambda x: x[1])[0]
    return {'reasoning': response, 'preferred_profile': preferred_profile}


@dataclass(frozen=True)
class Ranking:
    # Represents a 2way ranking between two profiles
    paper_text: str
    reasoning: str  # let the model generate the reasoning before returning the preferred profile
    profiles: tuple[Profile, Profile]
    preferred_profile: int  # (0 or 1) The index of the preferred profile in the profiles tuple

    def __str__(self) -> str:
        return f"""Paper Text:\n\n{self.paper_text}\n\n\nProfile 1: {self.profiles[0]}\n\n\nProfile 2: {self.profiles[1]}\n\n\nReasoning: {self.reasoning}\n\nPreferred Profile: {self.preferred_profile + 1}"""

    @staticmethod
    def parse_reasoning_json(obj: EvaluationResult) -> str:
        # fuzzy find the reasoning key in the json object
        # The json object should have the following structure:
        # {
        #     "reasoning": "[Your Reasoning]",
        #     "preferred_profile": [1 or 2]
        # }

        for key in obj:
            if 'reason' in key.lower():
                return obj[key]

        log(f'Invalid reasoning format: {obj}.', level=LogLevel.WARNING)
        return ''

    @staticmethod
    def parse_preferred_profile_json(obj: EvaluationResult) -> int:
        # Returns the index of the preferred profile (0 or 1) (0 if error)

        # fuzzy find the preferred_profile key in the json object
        # The json object should have the following structure:
        # {
        #     "reasoning": "[Your Reasoning]",
        #     "preferred_profile": [1 or 2]
        # }
        for key in obj:
            if 'preferred' in key.lower():
                number = obj[key]
                if number not in [0, 1, 2]:
                    log(f'Invalid preferred profile format: {obj}.', level=LogLevel.WARNING)
                    return 0

                return number - 1

        log(f'Invalid preferred profile format: {obj}.', level=LogLevel.WARNING)
        return 0

    @staticmethod
    def parse_reasoning(text: str) -> str:
        # Return the text between the first occurrence of 'Reasoning: ' and the next '\n\nPreferred Profile:'
        if 'reasoning:' not in text.lower():
            log(f'Invalid reasoning format: {text}.', level=LogLevel.WARNING)
            return ''

        # Ignore case when searching for 'reasoning:'
        return text.split('easoning:')[1].split('\n\nPreferred Profile:')[0]

    @staticmethod
    def parse_preferred_profile(text: str) -> bool:
        # Return the tuple (preferred profile, other profile) based on the text
        # Find the first number after 'Preferred Profile: ' and return the corresponding profile
        match = re.search(r'[P|p]referred [P|p]rofile: (\d)', text)
        if match:
            number = int(match.group(1))
            if number < 1 or number > 2:
                log(f'Invalid preferred profile format: {text}.', level=LogLevel.WARNING)
                return True

            return number == 1

        log(f'Invalid preferred profile format: {text}.', level=LogLevel.WARNING)
        return True

    @staticmethod
    def parse(text: str) -> Ranking:
        # Return the text between the first occurrence of 'Paper Text: ' and the next '\n\n\nProfile 1:'
        paper_text, rest_text = text.split('Paper Text:')[1].split('\n\n\nProfile 1:', maxsplit=1)

        profile_1, profile_2 = rest_text.split('\n\n\nProfile 2:', maxsplit=1)

        is_profile_1_preferred = Ranking.parse_preferred_profile(rest_text)

        return Ranking(
            paper_text=paper_text,
            reasoning=Ranking.parse_reasoning(rest_text),
            profiles=(Profile.parse(profile_1), Profile.parse(profile_2)),
            preferred_profile=0 if is_profile_1_preferred else 1,
        )


@dataclass(frozen=True)
class Combination:
    input_profiles: list[Profile]
    combined_profile: Profile

    def __str__(self) -> str:
        input_profiles = '\n\n'.join(str(profile) for profile in self.input_profiles)
        return f"""Input Profiles:\n{input_profiles}\n\n\nCombined Profile:\n{self.combined_profile}"""

    @staticmethod
    def parse(text: str) -> Combination:
        # Return the text between the first occurrence of 'Input Profiles:\n' and the next '\n\nCombined Profile:'
        input_profiles = text.split('Input Profiles:\n')[1].split('\n\nCombined Profile:')[0]

        # Return the text between the first occurrence of '\n\nCombined Profile:\n' and the end of the text
        combined_profile = text.split('\n\nCombined Profile:\n')[1]

        return Combination(
            input_profiles=[Profile.parse(profile) for profile in input_profiles.split('\n\n\n')],
            combined_profile=Profile.parse(combined_profile),
        )


DatabaseTypes = TypeVar('DatabaseTypes', Example, Summary, Combination, Ranking)


class Retriever(Protocol, Generic[DatabaseTypes]):
    def invoke(self, input: str) -> list[DatabaseTypes]:
        ...

    def batch(self, inputs: list[str]) -> list[list[DatabaseTypes]]:
        ...


class RetrieverGetter(Protocol):
    def __call__(self, return_type: Type[DatabaseTypes]) -> Retriever[DatabaseTypes]:
        ...
