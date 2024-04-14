from tqdm import tqdm
from pprint import pprint
from itertools import product

from src.instance import Instance, extract_from_abstracts, extract_from_full_texts, extract_from_summaries
from src.types import Profile


MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


MODELS = [
    # 'OpenAI/gpt-3.5-turbo',
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
]

NUMBER_OF_EXAMPLES = [0, 1, 2]

EXTRACTORS = [
    extract_from_abstracts,
    extract_from_summaries,
    extract_from_full_texts,
]


def process_author(name: str, number_of_papers: int = 5) -> tuple[list[Profile], list[str]]:
    profiles: list[Profile] = []
    titles: set[str] = set()

    for model, number_of_examples, good_or_bad_examples, extract_func in tqdm(
        product(MODELS, NUMBER_OF_EXAMPLES, [True, False], EXTRACTORS),
        desc='Processing different models and extractors',
    ):
        result = Instance(
            model,
            number_of_examples,
            good_or_bad_examples,
            extract_func,
        ).run_for_author(name, number_of_papers=number_of_papers)

        profiles.append(result.profile)
        titles.update(result.titles)

    return profiles, list(titles)


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 2:
        sys.exit('Usage: python -m src <command> <query>')

    if sys.argv[1] == 'author':
        pprint(process_author(sys.argv[2], number_of_papers=5))
