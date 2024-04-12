from __future__ import annotations
from dataclasses import dataclass
import re

__COMPETENCY_PATTERN = re.compile(r'- (.+?): (.+)')


@dataclass
class Competency:
    name: str
    description: str

    def __str__(self) -> str:
        return f'{self.name}: {self.description}'

    @staticmethod
    def parse(text: str) -> Competency | None:
        # Return a Competency object if the text matches the pattern '- [COMPETENCY]: [DESCRIPTION]'

        match = __COMPETENCY_PATTERN.match(text)
        if match:
            name, description = match.groups()
            return Competency(name=name, description=description)
        return None


@dataclass
class Profile:
    profile_summary: str
    competencies: list[Competency]

    def __str__(self) -> str:
        competencies = '\n'.join(f'- {c}' for c in self.competencies)
        return f"""Profile Summary: "{self.profile_summary}"

Competencies:
{competencies}
"""

    @staticmethod
    def _parse_profile_summary(text: str) -> str:
        # Return the text between the first occurrence of 'Profile Summary: "' and the next '"'
        assert 'Profile Summary: "' in text, 'Profile Summary not found in text'
        return text.split('Profile Summary: "')[1].split('"')[0]

    @staticmethod
    def _parse_competencies(text: str) -> list[Competency]:
        # Returns the list of competencies after the first occurrence of 'Competencies:\n' while the competencies are not empty and the line matches the pattern '- [COMPETENCY]: [DESCRIPTION]'
        assert 'Competencies:\n' in text, 'Competencies not found in text'

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
        return Profile(
            profile_summary=Profile._parse_profile_summary(text),
            competencies=Profile._parse_competencies(text),
        )


@dataclass
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
        # Return the text between the first occurrence of 'Abstract: "' and the next '"'
        assert 'Abstract: "' in text, 'Abstract not found in text'
        return text.split('Abstract: "')[1].split('"')[0]

    @staticmethod
    def parse(text: str) -> Example:
        return Example(
            abstract=Example._parse_abstract(text),
            profile=Profile.parse(text),
        )


@dataclass
class Query:
    abstract: str
    author: str
