import os

from dataclasses import dataclass
from tqdm import tqdm
from pprint import pprint
from itertools import product

from src.instance import Instance, extract_from_abstracts, extract_from_full_texts, extract_from_summaries
from src.types import Profile
from src.util import timeit

os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['OPENAI_BASE_URL'] = 'http://http://coder.aifb.kit.edu:8080/v1'

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

GOOD_OR_BAD_EXAMPLES = [True, False]


@dataclass
class ExtractedProfile:
    profile: Profile
    model: str
    number_of_examples: int
    good_or_bad_examples: bool
    extract_func: str


@dataclass
class AuthorExtractionResult:
    profiles: list[ExtractedProfile]
    titles: list[str]


@timeit('Processing Author')
def process_author(name: str, number_of_papers: int = 5) -> AuthorExtractionResult:
    profiles: list[ExtractedProfile] = []
    titles: set[str] = set()

    for model, number_of_examples, good_or_bad_examples, extract_func in tqdm(
        product(MODELS, NUMBER_OF_EXAMPLES, GOOD_OR_BAD_EXAMPLES, EXTRACTORS),
        desc='Processing different models and extractors',
    ):
        if not good_or_bad_examples and number_of_examples == 0:
            continue

        result = Instance(
            model,
            number_of_examples,
            good_or_bad_examples,
            extract_func,
        ).run_for_author(name, number_of_papers=number_of_papers)

        profiles.append(
            ExtractedProfile(
                profile=result.profile,
                model=model,
                number_of_examples=number_of_examples,
                good_or_bad_examples=good_or_bad_examples,
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
