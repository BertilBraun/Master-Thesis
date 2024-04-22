import os
import random
import src.openai_defines  # noqa # sets the OpenAI API key and base URL to the environment variables

from tqdm import tqdm
from itertools import product

from src.evaluation import evaluate_with
from src.instance import extract_from_abstracts, extract_from_full_texts, extract_from_summaries, run_query_for_instance
from src.database import add_element_to_database, database_size
from src.papers import get_papers_by_author, get_random_papers
from src.types import AuthorExtractionResult, ExtractedProfile, Profile, Example, ExampleType, Instance
from src.util import timeit
from src.log import LogLevel, datetime_str, log

# Remove base_url for OpenAI API and set the API key and use one of the following models to run the inference on the OpenAI API
# MODELS = [
#     'gpt-3.5-turbo',
#     'gpt-4-turbo',
#     'gpt-4',
# ]

EVALUATION_MODEL = 'neural'  # TODO should be something stronger like 'gpt-4-turbo'

MODELS = [
    # 'gpt-4',  # maps to Hermes-2-Pro-Mistral-7B.Q2_K via LocalAI
    'mistral',
    # 'neural',
    # 'mixtral',
]

EXAMPLES = [
    (ExampleType.POSITIVE, 2),
    (ExampleType.POSITIVE, 1),
    (ExampleType.POSITIVE, 0),
    # ExampleType.NEGATIVE currently not supported by chroma db
    # (ExampleType.NEGATIVE, 2),
    # (ExampleType.NEGATIVE, 1),
    # ExampleType.NEGATIVE with number_of_examples=0 does not make sense as it is the same as ExampleType.POSITIVE with number_of_examples=0
]

EXTRACTORS = [
    extract_from_abstracts,
    extract_from_summaries,
    extract_from_full_texts,
]


@timeit('Processing Author')
def process_author(name: str, number_of_papers: int = 5) -> AuthorExtractionResult:
    profiles: list[ExtractedProfile] = []

    query = get_papers_by_author(name, number_of_papers=number_of_papers)

    for model, (example_type, number_of_examples), extract_func in tqdm(
        product(MODELS, EXAMPLES, EXTRACTORS),
        desc='Processing different models and extractors',
    ):
        instance = Instance(
            model,
            number_of_examples,
            example_type,
            extract_func,
        )

        try:
            profile = run_query_for_instance(instance, query)
        except AssertionError as e:
            log(f'Error processing {instance=}', e, level=LogLevel.WARNING)
            continue

        profiles.append(ExtractedProfile(profile=profile, instance=instance))

    evaluation_result = evaluate_with(EVALUATION_MODEL, query, profiles)
    log('Final evaluation result:', evaluation_result, 'for author:', name)
    best_profile, best_score = evaluation_result[0]
    log('Best profile:', best_profile, 'with score:', best_score)

    return AuthorExtractionResult(
        profiles=profiles,
        titles=query.titles,
        author=query.author,
    )


def generate_example_references(number_of_references_to_generate: int):
    add_initial_example_references()

    # Use the actual OpenAI API not the LocalAI for generating as best results are expected from the largest models
    # src.openai_defines.BASE_URL_LLM = None

    # Get papers from different topics
    queries = get_random_papers(number_of_references_to_generate)

    os.makedirs('logs/generated_example_references/', exist_ok=True)

    generated_examples_file = f'logs/generated_example_references/{datetime_str()}.log'

    for query in queries:
        # Use one abstract at a time in a 1 shot prompt
        instance = Instance(
            'gpt-4',
            number_of_examples=1,
            example_type=ExampleType.POSITIVE,
            extract=extract_from_abstracts,
        )
        profile = run_query_for_instance(instance, query)

        example = Example(abstract=query.abstracts[0], profile=profile)

        # Write the extracted Profile as reference to a file
        add_element_to_database(example, is_reference=True)

        log(example, use_pprint=True, log_file_name=generated_examples_file)
        log('\n\n\n\n\n', log_file_name=generated_examples_file)


def add_initial_example_references():
    # Add some initial references to the database.
    if database_size(Example) > 0:
        return  # Do not add references if there are already references in the database

    add_element_to_database(
        Example(
            """A problem currently faced is the inability of an organisation to know the competences that the organisation masters, thereby bringing forth greater difficulties to the decision-making process, planning and team formation. In the scientific environment, this problem prejudices the multi-disciplinary research and communities creation. We propose a technique to create/suggest scientific web communities based on scientists' competences, identified using their scientific publications and considering that a possible indication for a person's participation in a community is her/his published knowledge and degree of expertise. The project also proposes an analysis structure providing an evolutionary visualisation of the virtual scientific community knowledge build-up.""",
            Profile.parse(
                """Domain: "Expert in developing web communities through competence analysis."

Competencies:
- Competence Identification: Utilizes scientific publications to map out individual competencies within an organization.
- Community Building: Develops web-based scientific communities by aligning similar expertises.
- Decision Support Systems: Enhances decision-making with structured competence visibility.
- Team Formation: Facilitates effective team assembly based on clearly identified competences.
- Knowledge Visualization: Implements evolutionary visual tools to represent the growth of virtual scientific communities.
- Expertise Analysis: Analyzes and suggests roles based on individuals’ published knowledge and expertise levels.""",
            ),
        ),
        is_reference=True,
    )

    add_element_to_database(
        Example(
            """Information extraction (IE) aims to extract structural knowledge (such as entities, relations, and events) from plain natural language texts. Recently, generative Large Language Models (LLMs) have demonstrated remarkable capabilities in text understanding and generation, allowing for generalization across various domains and tasks. As a result, numerous works have been proposed to harness abilities of LLMs and offer viable solutions for IE tasks based on a generative paradigm. To conduct a comprehensive systematic review and exploration of LLM efforts for IE tasks, in this study, we survey the most recent advancements in this field. We first present an extensive overview by categorizing these works in terms of various IE subtasks and learning paradigms, then we empirically analyze the most advanced methods and discover the emerging trend of IE tasks with LLMs. Based on thorough review conducted, we identify several insights in technique and promising research directions that deserve further exploration in future studies. We maintain a public repository and consistently update related resources at: https://github.com/quqxui/Awesome-LLM4IE-Papers.""",
            Profile.parse(
                """Domain: "Expert in enhancing information extraction with generative LLMs."

Competencies:
- Information Extraction Technologies: Specializes in using generative Large Language Models (LLMs) to structurally analyze text for entities, relations, and events.
- Domain Generalization: Applies LLMs across varied domains, demonstrating adaptability in text understanding and generation.
- Systematic Reviews: Conducts comprehensive analyses of current LLM applications in information extraction, identifying cutting-edge methods.
- Subtask Categorization: Classifies advancements in LLM-based IE by subtasks and learning paradigms, offering detailed insights.
- Emerging Trends Identification: Pinpoints and explores new directions in LLM applications for future information extraction tasks.
- Resource Sharing: Maintains and updates a public repository of significant research in LLM-enhanced information extraction."""
            ),
        ),
        is_reference=True,
    )


def format_mail(result: AuthorExtractionResult) -> str:
    # Formats the extraction result into a mail template for the author to request a evaluation of the extracted profiles

    titles = '- ' + '\n- '.join(result.titles)

    # Make a copy of the profiles list to shuffle it
    result_profiles = [profile for profile in result.profiles]

    # Shuffle the profiles list to avoid bias
    random.shuffle(result_profiles)

    profiles = ''
    for i, profile in enumerate(result_profiles):
        profiles += f'{i+1}.: {profile.profile}\n\n'

    return f"""
Hello Prof {result.author},

As part of our ongoing research, we are developing a system to match researchers with the most suitable scientific communities based on their competencies and expertise areas. We are developing a system that can automatically extract competencies and expertise areas from scientific papers and we would like to know how well you think the extracted profiles match your competencies and expertise areas.

We have processed the following papers for you:
{titles}

Based on these papers, we have extracted the following profiles for you:
{profiles}

We'd like to know which profile you think is the best match for you. Please provide a score between 0 and 100 for each profile based on how well you think it matches your competencies, themes, and expertise areas mentioned in the papers.

A format for your response could be:
1. 60
2. 80
3. 70   
4. 90
5. 85

Thank you for your time and we look forward to hearing from you soon.

Best regards,
Bertil Braun"""


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 2:
        sys.exit('Usage: python -m src <command> <query>')

    if sys.argv[1] == 'add' and sys.argv[2] == 'references':
        add_initial_example_references()

    if sys.argv[1] == 'gen':
        generate_example_references(int(sys.argv[2]))

    if sys.argv[1] == 'author':
        result = process_author(sys.argv[2], number_of_papers=5)

        log('Final result:', result, level=LogLevel.DEBUG)

        log(result, use_pprint=True)

        log('-' * 50)
        log(format_mail(result))
