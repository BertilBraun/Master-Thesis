from enum import Enum
import os
import re
import time
from dataclasses import dataclass

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import pipeline, TextGenerationPipeline


def datetime_str() -> str:
    return time.strftime('%Y-%m-%d %H %M %S')


def time_str() -> str:
    return time.strftime('%H:%M:%S')


class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


LOG_FILE = f'logs/log {datetime_str()}.txt'
LOG_LEVEL = LogLevel.INFO
os.makedirs('logs', exist_ok=True)
log_file = open(LOG_FILE, 'w')


def log(*args, level: LogLevel = LogLevel.INFO, **kwargs) -> None:
    timestamp = f'[{time_str()}]'
    print(timestamp, *args, **kwargs, file=log_file)
    if level.value >= LOG_LEVEL.value:
        print(timestamp, *args, **kwargs)


def timeit(message: str, level: LogLevel = LogLevel.INFO):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            log(f'{message}: {time.time() - start} seconds', level=level)
            return res

        return wrapper

    return decorator


@dataclass
class Competency:
    name: str
    description: str

    def __str__(self) -> str:
        return f'{self.name}: {self.description}'


@dataclass
class Profile:
    profile_summary: str
    competencies: list[Competency]


@dataclass
class Example:
    abstract: str
    profile: Profile

    def __str__(self) -> str:
        competencies = '\n'.join(f'- {c}' for c in self.profile.competencies)
        return f"""
Abstract:
{self.abstract}

Profile Summary: "{self.profile.profile_summary}"

Competencies:
{competencies}
"""


EXAMPLES = [
    Example(
        abstract='Through the application of deep learning techniques to satellite imagery, this research uncovers new patterns in urban development, contributing to more sustainable city planning.',
        profile=Profile(
            profile_summary='Expert in applying AI for sustainable urban development.',
            competencies=[
                Competency(
                    name='AI in Urban Planning',
                    description='Utilizes deep learning to analyze satellite images for city planning.',
                ),
                Competency(
                    name='Sustainable Development', description='Innovates in sustainable urban development strategies.'
                ),
                Competency(
                    name='Pattern Recognition', description='Identifies key urban development patterns using AI.'
                ),
                Competency(name='Data Analysis', description='Expert in analyzing large-scale geographical data.'),
            ],
        ),
    ),
    Example(
        abstract="Examining social media's impact on political discourse, this study employs natural language processing (NLP) to analyze sentiment and influence in online discussions, shedding light on digital communication's role in shaping public opinion.",
        profile=Profile(
            profile_summary='Specialist in digital communication and political discourse analysis.',
            competencies=[
                Competency(
                    name='NLP and Sentiment Analysis',
                    description='Applies NLP to understand social media influence.',
                ),
                Competency(
                    name='Digital Communication', description='Studies the impact of online platforms on communication.'
                ),
                Competency(
                    name='Public Opinion Research',
                    description='Analyzes how digital discourse shapes political opinions.',
                ),
                Competency(
                    name='Data-Driven Insights',
                    description='Generates insights into political discussions using data analysis.',
                ),
            ],
        ),
    ),
]


MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

generator: TextGenerationPipeline = pipeline('text-generation', model=MODEL)  # type: ignore


def parse_profile_summary(text: str) -> str:
    # Return the text between the first occurrence of 'Profile Summary: "' and the next '"'
    assert 'Profile Summary: "' in text, 'Profile Summary not found in text'
    return text.split('Profile Summary: "')[1].split('"')[0]


def parse_competencies(text: str) -> list[Competency]:
    # Returns the list of competencies after the first occurrence of 'Competencies:\n' while the competencies are not empty and the line matches the pattern '- [COMPETENCY]: [DESCRIPTION]'
    assert 'Competencies:\n' in text, 'Competencies not found in text'

    pattern = re.compile(r'- (.+?): (.+)')

    text = text.split('Competencies:\n')[1]

    competencies: list[Competency] = []
    for line in text.split('\n'):
        match = pattern.match(line)
        if match:
            name, description = match.groups()
            competencies.append(Competency(name=name, description=description))
        elif competencies:
            # If the line doesn't match the pattern and we have already found competencies, we break
            break

    return competencies


def parse_profile(text: str) -> Profile:
    return Profile(
        profile_summary=parse_profile_summary(text),
        competencies=parse_competencies(text),
    )


@timeit('Extracting competencies')
def extract(abstract: str, examples: list[Example]) -> Profile:
    examples_str = '\n\n\n'.join(f'Example {i + 1}:\n{e}' for i, e in enumerate(examples))

    # Define the prompt
    prompt = f"""Examples:

---
{examples_str}
---

Task Description:

Extract and summarize key competencies from scientific paper abstracts, aiming for a general overview suitable across disciplines. Begin with a concise profile summary that captures the main area of expertise in about ten words, abstract enough to apply broadly within a scientific context. Then, list three to eight specific competencies with brief descriptions based on the abstract.

The following is now your task. Please generate a profile summary and competencies based on the following abstract. Do not generate anything except the profile summary and competencies based on the abstract.

Abstract: "{abstract}"

"""

    # Generate the response
    response = generator(prompt, max_new_tokens=120, num_return_sequences=1)

    generated_text: str = response[0]['generated_text']  # type: ignore

    log(generated_text, level=LogLevel.DEBUG)

    generated_text = generated_text.replace(prompt, '')  # Remove the prompt from the generated text

    return parse_profile(generated_text)


res = extract(
    'Investigating the efficacy of machine learning algorithms in predicting stock market trends, this paper presents a model that outperforms traditional analytical methods.',
    EXAMPLES,
)

log(res)
