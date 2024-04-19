import os

os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['OPENAI_BASE_URL'] = 'http://coder.aifb.kit.edu:8080'

from tqdm import tqdm
from itertools import product

from src.evaluation import evaluate_with
from src.instance import extract_from_abstracts, extract_from_full_texts, extract_from_summaries, run_for_author
from src.database import DB
from src.papers import get_papers_by_author
from src.types import AuthorExtractionResult, ExtractedProfile, Profile, Example, ExampleType, Instance
from src.util import timeit
from src.log import LogLevel, log

# Remove base_url for OpenAI API and set the API key and use one of the following models to run the inference on the OpenAI API
# MODELS = [
#     'gpt-3.5-turbo',
#     'gpt-4-turbo',
#     'gpt-4',
# ]

EVALUATION_MODEL = 'neural'  # TODO should be something stronger like 'gpt-4-turbo'

MODELS = [
    'neural',
    'mistral',
    'mixtral',
]

EXAMPLES = [
    (ExampleType.POSITIVE, 2),
    (ExampleType.POSITIVE, 1),
    (ExampleType.POSITIVE, 0),
    # ExampleType.NEGATIVE currently not supported by langchain
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
            profile = run_for_author(instance, query)
        except Exception as e:
            log(f'Error processing {instance=}', e, level=LogLevel.WARNING)
            continue

        profiles.append(ExtractedProfile(profile=profile, instance=instance))

    evaluation_result = evaluate_with(EVALUATION_MODEL, query, profiles)
    log(evaluation_result)

    return AuthorExtractionResult(
        profiles=profiles,
        titles=query.titles,
        author=query.author,
    )


def add_initial_references():
    DB.add(
        Example(
            """A problem currently faced is the inability of an organisation to know the competences that the organisation masters, thereby bringing forth greater difficulties to the decision-making process, planning and team formation. In the scientific environment, this problem prejudices the multi-disciplinary research and communities creation. We propose a technique to create/suggest scientific web communities based on scientists' competences, identified using their scientific publications and considering that a possible indication for a person's participation in a community is her/his published knowledge and degree of expertise. The project also proposes an analysis structure providing an evolutionary visualisation of the virtual scientific community knowledge build-up.""",
            Profile.parse(
                """Profile Summary: "Expert in developing web communities through competence analysis."

Competencies:
- Competence Identification: Utilizes scientific publications to map out individual competencies within an organization.
- Community Building: Develops web-based scientific communities by aligning similar expertises.
- Decision Support Systems: Enhances decision-making with structured competence visibility.
- Team Formation: Facilitates effective team assembly based on clearly identified competences.
- Knowledge Visualization: Implements evolutionary visual tools to represent the growth of virtual scientific communities.
- Expertise Analysis: Analyzes and suggests roles based on individuals’ published knowledge and expertise levels.""",
            ),
        ),
        author='Sergio Rodrigues',
        is_reference=True,
    )

    DB.add(
        Example(
            """Information extraction (IE) aims to extract structural knowledge (such as entities, relations, and events) from plain natural language texts. Recently, generative Large Language Models (LLMs) have demonstrated remarkable capabilities in text understanding and generation, allowing for generalization across various domains and tasks. As a result, numerous works have been proposed to harness abilities of LLMs and offer viable solutions for IE tasks based on a generative paradigm. To conduct a comprehensive systematic review and exploration of LLM efforts for IE tasks, in this study, we survey the most recent advancements in this field. We first present an extensive overview by categorizing these works in terms of various IE subtasks and learning paradigms, then we empirically analyze the most advanced methods and discover the emerging trend of IE tasks with LLMs. Based on thorough review conducted, we identify several insights in technique and promising research directions that deserve further exploration in future studies. We maintain a public repository and consistently update related resources at: https://github.com/quqxui/Awesome-LLM4IE-Papers.""",
            Profile.parse(
                """Profile Summary: "Expert in enhancing information extraction with generative LLMs."

Competencies:
- Information Extraction Technologies: Specializes in using generative Large Language Models (LLMs) to structurally analyze text for entities, relations, and events.
- Domain Generalization: Applies LLMs across varied domains, demonstrating adaptability in text understanding and generation.
- Systematic Reviews: Conducts comprehensive analyses of current LLM applications in information extraction, identifying cutting-edge methods.
- Subtask Categorization: Classifies advancements in LLM-based IE by subtasks and learning paradigms, offering detailed insights.
- Emerging Trends Identification: Pinpoints and explores new directions in LLM applications for future information extraction tasks.
- Resource Sharing: Maintains and updates a public repository of significant research in LLM-enhanced information extraction."""
            ),
        ),
        author='Derong Xu',
        is_reference=True,
    )


if __name__ == '__main__':
    import sys
    from pprint import pprint

    if len(sys.argv) <= 2:
        sys.exit('Usage: python -m src <command> <query>')

    if sys.argv[1] == 'add' and sys.argv[2] == 'references':
        add_initial_references()

    if sys.argv[1] == 'author':
        pprint(process_author(sys.argv[2], number_of_papers=5))
