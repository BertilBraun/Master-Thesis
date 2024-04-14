from tqdm import tqdm
from pprint import pprint
from typing import Callable

from src.papers import get_authors_of_kit, get_papers_by_author
from src.gpt import query_openai, query_transformers
from src.db import DB
from src.util import timeit
from src.log import log, LogLevel
from src.types import Profile, Example, Query


MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


def generate_prompt(query: Query, examples: list[Example]) -> str:
    # Define the prompt
    prompt = f"""Task Description:

Extract and summarize key competencies from scientific paper abstracts, aiming for a general overview suitable across disciplines. Begin with a concise profile summary that captures the main area of expertise in about ten words, abstract enough to apply broadly within a scientific context. Then, list three to eight specific competencies with brief descriptions based on the abstract.

The following is now your task. Please generate a profile summary and competencies based on the following abstract. Do not generate anything except the profile summary and competencies based on the abstract.

Abstract: "{query.abstracts}"

"""
    if not examples:
        return prompt

    # If there are examples, add them to the start of the prompt
    examples_str = '\n\n'.join(f'Example {i + 1}:\n\n{e}' for i, e in enumerate(examples))

    return f"""Examples:

---
{examples_str}
---

{prompt}"""


def process_generated_text(generated_text: str, query: Query, is_reference: bool = False) -> Profile:
    # Parse the generated text into a Profile object
    profile = Profile.parse(generated_text)

    log('Profile:', profile, level=LogLevel.DEBUG)

    DB.add(
        Example(abstract=query.abstracts, profile=profile),
        query.author,
        is_reference=is_reference,
    )

    return profile


@timeit('Extracting competencies')
def extract(query: Query, examples: list[Example]) -> Profile:
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the abstract
    prompt = generate_prompt(query, examples)

    generated_text = query_transformers(prompt, model=MODEL)

    return process_generated_text(generated_text, query)


def extract_with_good_examples(query: Query, number_of_examples: int = 2) -> Profile:
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the abstract
    examples = DB.search(query.abstracts, limit=number_of_examples)

    return extract(query, examples)


def extract_with_bad_examples(query: Query, number_of_examples: int = 2) -> Profile:
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the abstract
    examples = DB.search_negative(query.abstracts, limit=number_of_examples)

    return extract(query, examples)


@timeit('Extracting competencies as reference')
def extract_as_reference(query: Query) -> None:
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the abstract
    examples = DB.search(query.abstracts, limit=2)

    prompt = generate_prompt(query, examples)

    generated_text = query_openai(prompt)

    process_generated_text(generated_text, query, is_reference=True)


def add_as_reference(abstract: str, profile: Profile, author: str) -> None:
    print(f'Adding reference of {author}.')
    DB.add(
        Example(abstract=abstract, profile=profile),
        author,
        rating=100,
        is_reference=True,
    )
    print('Reference added.')


def add_initial_references():
    # Warning: This will clear the database and then readd ONLY the initial references

    DB.clear()

    add_as_reference(
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
        author='Sergio Rodrigues',
    )

    add_as_reference(
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
        author='Derong Xu',
    )


def add_references():
    authors = get_authors_of_kit(10)

    for author in authors:
        print(f'Adding reference of {author.name}.')
        for paper in tqdm(get_papers_by_author(author.name), desc=author.name):
            extract_as_reference(paper)
        print('Reference added.')


def combine_profiles(profiles: list[Profile]) -> Profile:
    # TODO Combine multiple profiles into one using a LLM
    return profiles[0]


def process_author(
    name: str,
    number_of_papers: int,
    extract_func: Callable[[Query], Profile],
    combine_func: Callable[[list[Profile]], Profile],
) -> Profile:
    papers = get_papers_by_author(name, number_of_papers=number_of_papers)

    profiles: list[Profile] = []

    for paper in tqdm(papers, desc='Extracting from Abstract'):
        try:
            profile = extract_func(paper)
            profiles.append(profile)
        except AssertionError as e:
            print(f'Failed to extract Profile for: "{paper}" with error: {e}')

    return combine_func(profiles)


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 2:
        sys.exit('Usage: python -m src <command> <query>')

    if sys.argv[1] == 'db':
        pprint(DB.search(sys.argv[2]))

    if sys.argv[1] == 'good':
        pprint(extract_with_good_examples(Query(abstracts=sys.argv[2], author='user')))

    if sys.argv[1] == 'bad':
        pprint(extract_with_bad_examples(Query(abstracts=sys.argv[2], author='user')))

    if sys.argv[1] == 'ref':
        # add_initial_references()
        add_references()

    if sys.argv[1] == 'author':
        pprint(
            process_author(
                sys.argv[2],
                number_of_papers=5,
                extract_func=extract_with_good_examples,
                combine_func=combine_profiles,
            )
        )
