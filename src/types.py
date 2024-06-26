from __future__ import annotations

import re

from typing import Callable, Generic, Literal, Protocol, Type, TypeVar, TypedDict, overload
from dataclasses import dataclass, field

from openai.types.chat import ChatCompletionMessageParam

from src.log import LogLevel, log

_COMPETENCY_PATTERN = re.compile(r'([-|\d+].*?)? (.+?): (.+)')


@dataclass(frozen=True)
class Competency:
    name: str
    description: str

    def __str__(self) -> str:
        return f'{self.name}: {self.description}'

    @staticmethod
    def parse(text: str) -> Competency:
        # Return a Competency object if the text matches the pattern '- [COMPETENCY]: [DESCRIPTION]'

        match = _COMPETENCY_PATTERN.match(text)
        if match:
            prefix, name, description = match.groups()
            return Competency(name=name, description=description)
        log(f'Invalid competency format: {text}.', level=LogLevel.DEBUG)
        return Competency(name=f'Invalid competency format: {text}.', description='')


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

    def to_json(self) -> str:
        return (
            '{\n    "domain": "'
            + self.domain
            + '",\n    "competencies": {\n'
            + ',\n'.join(
                [
                    f'        "{competency.name}": "{competency.description}"'
                    for competency in self.competencies
                    if competency.name and competency.description
                ]
            )
            + '\n    }\n}'
        )

    @staticmethod
    def from_json(data: dict) -> Profile:
        return Profile(
            domain=data['domain'],
            competencies=[
                Competency(name=competency['name'], description=competency['description'])
                for competency in data['competencies']
            ],
        )

    @staticmethod
    def _parse_domain(text: str) -> str:
        # Return the text between the first occurrence of 'Domain:' and the next '\n'
        assert 'Domain:' in text, f'Domain not found in text: {text}'
        return text.split('Domain:')[1].split('\n')[0].strip().strip('"').strip()

    @staticmethod
    def _parse_competencies(text: str) -> list[Competency]:
        # Returns the list of competencies after the first occurrence of 'Competencies:\n' while the competencies are not empty and the line matches the pattern '- [COMPETENCY]: [DESCRIPTION]'
        text = text.replace('\n\n', '\n').replace('**', '')
        assert 'Competencies:\n' in text, f'Competencies not found in text: {text}'

        text = text.split('Competencies:\n')[1]

        competencies: list[Competency] = []
        for line in text.split('\n'):
            competency = Competency.parse(line)
            if competency.name.strip() and competency.description.strip():
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

    @staticmethod
    def parse_json(obj: dict) -> Profile:
        # fuzzy find the domain and competencies keys in the json object
        # The json object should have the following structure:
        # {
        #     "domain": "[Short Domain Description]",
        #     "competencies": {
        #         "[Competency 1]": "[Detailed explanation of how Competency 1 is demonstrated in the text]",
        #         "[Competency 2]": "[Detailed explanation of how Competency 2 is demonstrated in the text]",
        #         ...
        #     }
        # }

        domain = 'Domain not found'
        competencies: list[Competency] = []

        for key in obj:
            if 'domain' in key.lower():
                domain = obj[key]
            elif 'competencies' in key.lower():
                for competency in obj[key]:
                    if competency.strip():  # check if the competency is not empty
                        competencies.append(Competency(name=competency.strip(), description=str(obj[key][competency])))

        return Profile(domain=domain, competencies=competencies)


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


class Preference(TypedDict):
    prompt: str
    chosen: str
    rejected: str


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
        # Returns the index of the preferred profile (0 if error)

        # fuzzy find the preferred_profile key in the json object
        # The json object should have the following structure:
        # {
        #     "reasoning": "[Your Reasoning]",
        #     "preferred_profile": [1 or 2]
        # }
        for key in obj:
            if 'preferred' in key.lower():
                number = obj[key]
                if number < 1 or number > 2:
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

    def __add__(self, other: Query) -> Query:
        assert self.author == other.author, 'Authors must be the same to combine queries'
        return Query(
            full_texts=self.full_texts + other.full_texts,
            abstracts=self.abstracts + other.abstracts,
            titles=self.titles + other.titles,
            author=self.author,
        )


@dataclass(frozen=True)
class ExtractionResult:
    profile: Profile
    titles: list[str]
    author: str


DatabaseTypes = TypeVar('DatabaseTypes', Example, Summary, Combination, Ranking)


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
    debug_context_name: str

    def __init__(self, model: str, debug_context_name: str = ''):
        ...

    @overload
    def batch(
        self,
        prompts: list[list[Message]],
        /,
        stop: list[str] = [],
        response_format: Literal['text'] = 'text',
        temperature: float = 0.5,
    ) -> list[str]:
        ...

    @overload
    def batch(
        self,
        prompts: list[list[Message]],
        /,
        stop: list[str] = [],
        response_format: Literal['json_object'] = 'json_object',
        temperature: float = 0.5,
    ) -> list[dict]:
        ...

    @overload
    def invoke(
        self,
        prompt: list[Message],
        /,
        stop: list[str] = [],
        response_format: Literal['text'] = 'text',
        temperature: float = 0.5,
    ) -> str:
        ...

    @overload
    def invoke(
        self,
        prompt: list[Message],
        /,
        stop: list[str] = [],
        response_format: Literal['json_object'] = 'json_object',
        temperature: float = 0.5,
    ) -> dict:
        ...

    def invoke_profile_custom(self, prompt: list[Message]) -> Profile:
        ...

    def invoke_profile_json(self, prompt: list[Message]) -> Profile:
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
    extract: Callable[[Query, RetrieverGetter, LanguageModel], Profile]

    @staticmethod
    def empty_instance() -> Instance:
        return Instance(
            model='',
            number_of_examples=0,
            extract=lambda q, r, l: Profile(domain='', competencies=[]),  # noqa
        )

    def __str__(self) -> str:
        return f'Instance({self.model} ({self.number_of_examples} examples) - {self.extract.__name__})'


@dataclass(frozen=True)
class ExtractedProfile:
    profile: Profile
    model: str  # Identifier from OpenAI/Insomnium
    number_of_examples: int
    extraction_function: str
    extraction_time: float

    @staticmethod
    def from_profile(profile: Profile) -> ExtractedProfile:
        return ExtractedProfile(
            profile=profile,
            model='None',
            number_of_examples=0,
            extraction_function='None',
            extraction_time=0.0,
        )


@dataclass(frozen=True)
class RankingResult:
    # Represents a 2way ranking between two profiles
    profiles: tuple[int, int]
    preferred_profile_index: int  # (0 or 1) The index of the preferred profile in the profiles tuple
    reasoning: str | None  # let the model generate the reasoning before returning the preferred profile

    @property
    def winner(self) -> int:
        return self.profiles[self.preferred_profile_index]

    @property
    def loser(self) -> int:
        return self.profiles[1 - self.preferred_profile_index]


@dataclass(frozen=True)
class TournamentNode:
    match: RankingResult
    children: list[TournamentNode] = field(default_factory=list)

    @property
    def all_nodes(self) -> list[TournamentNode]:
        nodes: list[TournamentNode] = [self]
        for child in self.children:
            nodes.extend(child.all_nodes)
        return nodes

    @property
    def all_loser_nodes(self) -> list[TournamentNode]:
        if not self.children:
            return []
        if self.match.profiles[0] == self.match.profiles[1]:
            # If the profiles are the same, return the children of the first child
            return self.children[0].all_loser_nodes
        return self.children[1 - self.match.preferred_profile_index].all_nodes


@dataclass(frozen=True)
class AuthorResult:
    tournament: TournamentNode
    profiles: dict[int, ExtractedProfile]
    titles: list[str]
    author: str

    @staticmethod
    def from_json(data: dict) -> AuthorResult:
        tournament, profiles = parse_tournament_and_profiles_from_json(data)
        return AuthorResult(
            tournament=tournament,
            profiles=profiles,
            titles=data['titles'],
            author=data['author'],
        )


def parse_tournament_and_profiles_from_json(data: dict) -> tuple[TournamentNode, dict[int, ExtractedProfile]]:
    # convert tournament dictionary to TournamentNode
    def process_node(node_data: dict) -> TournamentNode:
        node = TournamentNode(
            match=RankingResult(
                profiles=node_data['match']['profiles'],
                preferred_profile_index=node_data['match']['preferred_profile_index'],
                reasoning=node_data['match'].get('reasoning', None),
            )
        )
        for child_data in node_data['children']:
            node.children.append(process_node(child_data))
        return node

    tournament = process_node(data['tournament'])

    # convert profiles dictionary to dict[int, ExtractedProfile]
    profiles = {}
    for key, profile_data in data['profiles'].items():
        profiles[int(key)] = ExtractedProfile(
            profile=Profile(
                domain=profile_data['profile']['domain'],
                competencies=[
                    Competency(competency['name'], competency['description'])
                    for competency in profile_data['profile']['competencies']
                ],
            ),
            model=profile_data['model'],
            number_of_examples=profile_data['number_of_examples'],
            extraction_function=profile_data['extraction_function'],
            extraction_time=profile_data['extraction_time'],
        )

    return tournament, profiles
