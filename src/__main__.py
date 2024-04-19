import os

os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['OPENAI_BASE_URL'] = 'http://coder.aifb.kit.edu:8080'

# from langchain.globals import set_debug
# set_debug(True)

from tqdm import tqdm
from pprint import pprint
from itertools import product
from dataclasses import dataclass

from src.instance import ExampleType, Instance, extract_from_abstracts, extract_from_full_texts, extract_from_summaries
from src.types import Profile
from src.util import timeit
from src.log import LogLevel, log

# Remove base_url for OpenAI API and set the API key and use one of the following models to run the inference on the OpenAI API
# MODELS = [
#     'gpt-3.5-turbo',
#     'gpt-4-turbo',
#     'gpt-4',
# ]

MODELS = [
    'mistral',
    'mixtral',
    'neural',
]

NUMBER_OF_EXAMPLES = [0, 1, 2]

EXTRACTORS = [
    extract_from_abstracts,
    extract_from_summaries,
    extract_from_full_texts,
]

EXAMPLE_TYPES = [
    ExampleType.POSITIVE,
    # ExampleType.NEGATIVE,
]


@dataclass
class ExtractedProfile:
    profile: Profile
    model: str
    number_of_examples: int
    example_type: ExampleType
    extract_func: str


@dataclass
class AuthorExtractionResult:
    profiles: list[ExtractedProfile]
    titles: list[str]


@timeit('Processing Author')
def process_author(name: str, number_of_papers: int = 5) -> AuthorExtractionResult:
    profiles: list[ExtractedProfile] = []
    titles: set[str] = set()

    for model, number_of_examples, example_type, extract_func in tqdm(
        product(MODELS, NUMBER_OF_EXAMPLES, EXAMPLE_TYPES, EXTRACTORS),
        desc='Processing different models and extractors',
    ):
        if example_type == ExampleType.NEGATIVE and number_of_examples == 0:
            continue

        instance = Instance(
            model,
            number_of_examples,
            example_type,
            extract_func,
        )

        try:
            result = instance.run_for_author(name, number_of_papers=number_of_papers)
        except Exception as e:
            log(
                f'Error processing {model=}, {number_of_examples=}, {example_type=}, {extract_func=}',
                e,
                level=LogLevel.WARNING,
            )
            continue

        profiles.append(
            ExtractedProfile(
                profile=result.profile,
                model=model,
                number_of_examples=number_of_examples,
                example_type=example_type,
                extract_func=extract_func.__name__,
            )
        )
        titles.update(result.titles)

    return AuthorExtractionResult(
        profiles=profiles,
        titles=list(titles),
    )


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 2:
        sys.exit('Usage: python -m src <command> <query>')

    if sys.argv[1] == 'author':
        pprint(process_author(sys.argv[2], number_of_papers=5))
