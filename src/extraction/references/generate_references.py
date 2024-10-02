from src.logic.language_model import OpenAILanguageModel
from src.extraction.evaluation import tournament_ranking
from src.extraction.extraction_custom import _extract_from_full_texts_custom, extract_from_abstracts_custom
from src.logic.database import (
    add_element_to_database,
    get_retriever_getter,
    get_sample_from_database,
)
from src.logic.papers import get_random_english_authors_abstracts
from src.logic.types import Combination, DatabaseTypes, ExtractedProfile, Example, Instance, Query, Ranking
from src.util.log import datetime_str, log


def __write_reference(element: DatabaseTypes, file_name: str) -> None:
    add_element_to_database(element, is_reference=True)

    log(element, use_pprint=True, log_file_name=file_name)


def generate_example_references(
    number_of_references_to_generate: int,
    reference_generation_model: str,
):
    from src.extraction.process_authors import run_query_for_instance

    # from src.initial_references import add_initial_example_references
    # add_initial_example_references()

    # Use the actual OpenAI API not the LocalAI for generating as best results are expected from the largest models
    # TODO src.openai_defines.BASE_URL_LLM = None

    # Get papers from different topics
    generated_examples_file = f'logs/generated_example_references/{datetime_str()}.log'

    for query in get_random_english_authors_abstracts(number_of_references_to_generate, number_of_papers_per_author=2):
        # Use one abstract at a time in a 1 shot prompt
        instance = Instance(
            reference_generation_model,
            number_of_examples=1,
            extract=extract_from_abstracts_custom,
        )

        if (profile := run_query_for_instance(instance, query)) is None:
            continue

        abstracts = '\n\n'.join(f'Abstract {i + 1}:\n{abstract}' for i, abstract in enumerate(query.abstracts))
        example = Example(abstracts=abstracts, profile=profile)

        # Write the extracted Profile as reference to a file and database
        __write_reference(example, generated_examples_file)


def generate_combination_references(
    number_of_references_to_generate: int,
    reference_generation_model: str,
):
    # from src.initial_references import add_initial_combination_references
    # add_initial_combination_references()

    # Use the actual OpenAI API not the LocalAI for generating as best results are expected from the largest models
    # TODO src.openai_defines.BASE_URL_LLM = None

    generated_combinations_file = f'logs/generated_combination_references/{datetime_str()}.log'

    for query in get_random_english_authors_abstracts(number_of_references_to_generate, number_of_papers_per_author=2):
        extracted_profiles, combined_profile = _extract_from_full_texts_custom(
            query,
            get_retriever_getter(max_number_to_retrieve=1),
            OpenAILanguageModel(model=reference_generation_model),
        )

        combination = Combination(
            input_profiles=extracted_profiles,
            combined_profile=combined_profile,
        )

        # Write the combination as reference to a file and database
        __write_reference(combination, generated_combinations_file)


def generate_ranking_references(
    number_of_references_to_generate: int,
    evaluation_model: str,
    other_reference_generation_model: str,
):
    from src.extraction.process_authors import run_query_for_instance

    # from src.initial_references import add_initial_ranking_references
    # add_initial_ranking_references()

    # Use the actual OpenAI API not the LocalAI for generating as best results are expected from the largest models
    # TODO src.openai_defines.BASE_URL_LLM = None

    generated_rankings_file = f'logs/generated_ranking_references/{datetime_str()}.log'

    for example in get_sample_from_database(Example, number_of_references_to_generate):
        query = Query(
            full_texts=['Unknown'],
            abstracts=[example.abstracts],
            titles=['Unknown'],
            author='Unknown',
        )
        other_profile = run_query_for_instance(
            Instance(
                model=other_reference_generation_model,
                number_of_examples=1,
                extract=extract_from_abstracts_custom,
            ),
            query,
        )

        if other_profile is None:
            continue

        all_profiles = {
            1: ExtractedProfile.from_profile(example.profile),
            2: ExtractedProfile.from_profile(other_profile),
        }

        root = tournament_ranking(evaluation_model, query, all_profiles)

        ranking = Ranking(
            paper_text=example.abstracts,
            reasoning=root.match.reasoning or 'No reasoning provided',
            profiles=(
                all_profiles[root.match.profiles[0]].profile,
                all_profiles[root.match.profiles[1]].profile,
            ),
            preferred_profile=root.match.preferred_profile_index,
        )

        # Write the evaluation as reference to a file and database
        __write_reference(ranking, generated_rankings_file)
