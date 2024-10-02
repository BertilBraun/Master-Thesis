from __future__ import annotations

import re

from dataclasses import dataclass


from src.util.log import LogLevel, log

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
            name = (
                name.strip()
                .replace('**', '')
                .replace('"', '')
                .replace('“', '')
                .replace('”', '')
                .replace('’', '')
                .replace('‘', '')
                .replace('[', '')
                .replace(']', '')
                .replace('(', '')
                .replace(')', '')
                .replace('—', '')
                .replace('–', '')
                .replace('•', '')
                .replace('·', '')
            )
            return Competency(name=name, description=description)
        log(f'Invalid competency format: {text}.', level=LogLevel.DEBUG)
        return Competency(name=text, description='')


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
        text = text.replace('\n\n', '\n').replace('**', '')
        assert 'Domain:' in text, f'Domain not found in text: {text}'
        return text.split('Domain:')[1].split('\n')[0].strip().strip('"').strip()

    @staticmethod
    def _parse_competencies(text: str) -> list[Competency]:
        # Returns the list of competencies after the first occurrence of 'Competencies:\n' while the competencies are not empty and the line matches the pattern '- [COMPETENCY]: [DESCRIPTION]'
        text = text.replace('\n\n', '\n').replace('**', '')
        assert 'Domain:' in text, f'Competencies not found in text: {text}'

        competencies: list[Competency] = []
        # Find the first line after 'Domain:.*?\n'
        for line in text.split('Domain:')[1].split('\n')[1:]:
            competency = Competency.parse(line)
            name_len = len(competency.name.strip())
            if 6 < name_len < 150 and not competency.name.strip().startswith('Competencies'):
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
