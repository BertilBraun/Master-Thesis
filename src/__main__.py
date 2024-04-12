import os
from pprint import pprint

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
    # TODO some stop at criteria '\n\n\n' or something
    response = generator(prompt, max_new_tokens=200, num_return_sequences=1)

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


if __name__ == '__main__':
    import sys

    # add_initial_references()

    if len(sys.argv) <= 2:
        sys.exit('Usage: python -m src <command> <query>')

    if sys.argv[1] == 'db':
        pprint(DB.search(sys.argv[2]))

    if sys.argv[1] == 'good':
        print('Extracting with good examples for:', sys.argv[2])
        pprint(extract_with_good_examples(Query(abstract=sys.argv[2], author='user')))

    if sys.argv[1] == 'bad':
        pprint(extract_with_bad_examples(Query(abstract=sys.argv[2], author='user')))
