import random


from src.language_model import OpenAILanguageModel
from src.evaluation import tournament_ranking
from src.extraction_custom import _extract_from_full_texts_custom
from src.extraction_json import extract_from_abstracts_json
from src.database import (
    add_element_to_database,
    get_retriever_getter,
    get_sample_from_database,
)
from src.papers import (
    get_authors_of_kit,
    get_papers_by_author,
    get_random_papers,
)
from src.types import (
    Combination,
    DatabaseTypes,
    ExtractedProfile,
    Example,
    Instance,
    Query,
    Ranking,
)
from src.log import LogLevel, datetime_str, log


OTHER_REFERENCE_GENERATION_MODEL = 'mistral'  # should be something other than the REFERENCE_GENERATION_MODEL to generate different results which can be used for ranking
REFERENCE_GENERATION_MODEL = 'neural'  # TODO should be something stronger like 'gpt-4-turbo'
EVALUATION_MODEL = 'neural'  # TODO should be something stronger like 'gpt-4-turbo'

OTHER_REFERENCE_GENERATION_MODEL = 'alias-fast'
REFERENCE_GENERATION_MODEL = 'alias-large'
EVALUATION_MODEL = 'alias-large'


def write_reference(element: DatabaseTypes, file_name: str) -> None:
    add_element_to_database(element, is_reference=True)

    log(element, use_pprint=True, log_file_name=file_name)
    log('\n\n\n\n\n', log_file_name=file_name)


def generate_example_references(number_of_references_to_generate: int, reference_generation_model: str):
    from src.initial_references import add_initial_example_references
    from src.__main__ import run_query_for_instance

    add_initial_example_references()

    # Use the actual OpenAI API not the LocalAI for generating as best results are expected from the largest models
    # TODO src.openai_defines.BASE_URL_LLM = None

    # Get papers from different topics
    queries = get_random_papers(number_of_references_to_generate)

    generated_examples_file = f'logs/generated_example_references/{datetime_str()}.log'

    for query in queries:
        # Use one abstract at a time in a 1 shot prompt
        instance = Instance(
            reference_generation_model,
            number_of_examples=1,
            extract=extract_from_abstracts_json,
        )

        if (profile := run_query_for_instance(instance, query)) is None:
            continue

        example = Example(abstract=query.abstracts[0], profile=profile)

        # Write the extracted Profile as reference to a file and database
        write_reference(example, generated_examples_file)


def generate_combination_references(number_of_references_to_generate: int):
    from src.initial_references import add_initial_combination_references

    add_initial_combination_references()

    # Use the actual OpenAI API not the LocalAI for generating as best results are expected from the largest models
    # TODO src.openai_defines.BASE_URL_LLM = None

    # Get a random subset of authors to generate the references from
    authors = get_authors_of_kit(number_of_references_to_generate * 2)
    random.shuffle(authors)
    authors = authors[:number_of_references_to_generate]

    queries = [get_papers_by_author(author.name, number_of_papers=2) for author in authors]
    queries = [
        Query(
            abstracts=query.abstracts,
            # Get papers from different authors but only look at the abstracts
            full_texts=query.abstracts,
            titles=query.titles,
            author=query.author,
        )
        for query in queries
    ]

    generated_combinations_file = f'logs/generated_combination_references/{datetime_str()}.log'
    generated_examples_file = f'logs/generated_example_references/{datetime_str()}.log'

    for query in queries:
        extracted_profiles, combined_profile = _extract_from_full_texts_custom(
            query,
            get_retriever_getter(max_number_to_retrieve=1),
            OpenAILanguageModel(model=REFERENCE_GENERATION_MODEL),
        )

        combination = Combination(
            input_profiles=extracted_profiles,
            combined_profile=combined_profile,
        )

        # Write the combination as reference to a file and database
        write_reference(combination, generated_combinations_file)

        if len(query.abstracts) != len(extracted_profiles):
            log(
                f'Number of abstracts ({len(query.abstracts)}) and extracted profiles ({len(extracted_profiles)}) do not match for query: {query}',
                level=LogLevel.WARNING,
            )
            continue

        for abstract, profile in zip(query.abstracts, extracted_profiles):
            example = Example(abstract=abstract, profile=profile)
            write_reference(example, generated_examples_file)


def generate_ranking_references(
    number_of_references_to_generate: int, evaluation_model: str, other_reference_generation_model: str
):
    # from src.initial_references import add_initial_ranking_references
    from src.__main__ import run_query_for_instance
    # add_initial_ranking_references()

    # Use the actual OpenAI API not the LocalAI for generating as best results are expected from the largest models
    # TODO src.openai_defines.BASE_URL_LLM = None

    generated_examples_file = f'logs/generated_example_references/{datetime_str()}.log'
    generated_rankings_file = f'logs/generated_ranking_references/{datetime_str()}.log'

    examples = get_sample_from_database(Example, number_of_references_to_generate)
    queries = [
        Query(full_texts=['Unknown'], abstracts=[example.abstract], titles=['Unknown'], author='Unknown')
        for example in examples
    ]
    other_profiles = [
        run_query_for_instance(
            Instance(model=other_reference_generation_model, number_of_examples=0, extract=extract_from_abstracts_json),
            query,
        )
        for query in queries
    ]
    for example, profile in zip(examples, other_profiles):
        if profile is not None:
            # Write the extracted Profile as reference to a file and database
            write_reference(Example(abstract=example.abstract, profile=profile), generated_examples_file)

    for example, query, other_profile in zip(examples, queries, other_profiles):
        if other_profile is None:
            continue

        all_profiles = {
            1: ExtractedProfile.from_profile(example.profile),
            2: ExtractedProfile.from_profile(other_profile),
        }

        root = tournament_ranking(evaluation_model, query, all_profiles)

        ranking = Ranking(
            paper_text=example.abstract,
            reasoning=root.match.reasoning,
            profiles=(all_profiles[root.match.profiles[0]].profile, all_profiles[root.match.profiles[1]].profile),
            preferred_profile=root.match.preferred_profile_index,
        )

        # Write the evaluation as reference to a file and database
        write_reference(ranking, generated_rankings_file)
