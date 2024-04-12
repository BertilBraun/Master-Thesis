import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import pipeline, TextGenerationPipeline

from src.db import DB
from src.util import timeit
from src.logging import log, LogLevel
from src.types import Profile, Competency, Example, Query

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


@timeit('Extracting competencies')
def extract(query: Query, examples: list[Example]) -> Profile:
    examples_str = '\n\n\n'.join(f'Example {i + 1}:\n\n{e}' for i, e in enumerate(examples))

    # Define the prompt
    prompt = f"""Examples:

---
{examples_str}
---

Task Description:

Extract and summarize key competencies from scientific paper abstracts, aiming for a general overview suitable across disciplines. Begin with a concise profile summary that captures the main area of expertise in about ten words, abstract enough to apply broadly within a scientific context. Then, list three to eight specific competencies with brief descriptions based on the abstract.

The following is now your task. Please generate a profile summary and competencies based on the following abstract. Do not generate anything except the profile summary and competencies based on the abstract.

Abstract: "{query.abstract}"

"""

    # Generate the response
    response = generator(prompt, max_new_tokens=120, num_return_sequences=1)

    generated_text: str = response[0]['generated_text']  # type: ignore

    log(generated_text, level=LogLevel.DEBUG)

    generated_text = generated_text.replace(prompt, '')  # Remove the prompt from the generated text

    profile = Profile.parse(generated_text)

    DB.add(
        Example(abstract=query.abstract, profile=profile),
        query.author,
        is_reference=False,
    )

    return profile


def extract_with_good_examples(query: Query) -> Profile:
    examples = DB.search(query.abstract, limit=2)

    return extract(query, examples)


def extract_with_bad_examples(query: Query) -> Profile:
    examples = DB.search_negative(query.abstract, limit=2)

    return extract(query, examples)


def add_as_reference(abstract: str, profile: Profile, author: str) -> None:
    DB.add(
        Example(abstract=abstract, profile=profile),
        author,
        rating=100,
        is_reference=True,
    )


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 2:
        sys.exit('Usage: python -m src <command> <query>')

    if sys.argv[1] == 'db':
        print(DB.search(sys.argv[2]))

    if sys.argv[1] == 'good':
        print(extract_with_good_examples(Query(abstract=sys.argv[2], author='user')))

    if sys.argv[1] == 'bad':
        print(extract_with_bad_examples(Query(abstract=sys.argv[2], author='user')))
