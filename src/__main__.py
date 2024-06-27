from itertools import product

from src.language_model import OpenAILanguageModel
from src.evaluation import tournament_ranking
from src.extraction_custom import (
    extract_from_abstracts_custom,
    extract_from_full_texts_custom,
    extract_from_summaries_custom,
)
from src.extraction_json import (
    extract_from_abstracts_json,
    extract_from_full_texts_json,
    extract_from_summaries_json,
)
from src.database import get_retriever_getter
from src.display import (
    dump_author_result_to_json,
    generate_html_file_for_tournament_evaluation,
    generate_html_file_for_tournament_ranking_result,
)
from src.papers import get_papers_by_author
from src.types import (
    AuthorResult,
    ExtractedProfile,
    Profile,
    Instance,
    Query,
)
from src.util import timeblock, timeit
from src.log import LogLevel, datetime_str, log


OTHER_REFERENCE_GENERATION_MODEL = 'mistral'  # should be something other than the REFERENCE_GENERATION_MODEL to generate different results which can be used for ranking
REFERENCE_GENERATION_MODEL = 'neural'  # TODO should be something stronger like 'gpt-4-turbo'
EVALUATION_MODEL = 'neural'  # TODO should be something stronger like 'gpt-4-turbo'

OTHER_REFERENCE_GENERATION_MODEL = 'alias-fast-instruct'
REFERENCE_GENERATION_MODEL = 'alias-fast-instruct'
EVALUATION_MODEL = 'alias-large-instruct'

DO_SHUFFLE_DURING_EVALUATION = True

MODELS = [
    # 'dev-phi-3-mini-128k',
    'dev-phi-3-medium',
    'dev-gemma-large',
    'dev-gemma-small',
    'dev-llama-3-large',
    'dev-llama-3-small',
    'alias-large-instruct',
    'alias-fast-instruct',
    # 'alias-fast',
    # TODO sind gut 'alias-fast-instruct',
    # TODO sind gut 'alias-large-instruct',
    # 'dev-llama-3-large',
    # 'mistral',
    # 'neural',
    # 'mixtral',
    # TODO 'phi3 mini' 3.8B parameters
    # TODO 'llama3' 8B parameters (ultra sota)
    # Set src.openai_defines.BASE_URL_LLM = None for and set the API key and use one of the following models to run the inference on the OpenAI API
    # TODO 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'
]

EXAMPLES = [1]  # TODO set to [1, 0]

EXTRACTORS = [
    extract_from_abstracts_custom,
    extract_from_abstracts_json,
    extract_from_summaries_custom,
    extract_from_summaries_json,
    extract_from_full_texts_custom,
    extract_from_full_texts_json,
][::2]  # Only use the custom extractors for now, as they seem to return better results
# [1::2]  # Only use the json extractors for now, as they are more reliable


@timeit('Processing Author')
def process_author(name: str, number_of_papers: int = 5) -> AuthorResult:
    log(f'Processing Author: {name=} {number_of_papers=}')
    profiles: dict[int, ExtractedProfile] = {}

    query = get_papers_by_author(name, number_of_papers=number_of_papers, KIT_only=True)

    extracted_profile_log = f'logs/extracted_profiles/{name}_{datetime_str()}.log'

    with timeblock(f'Processing Extracted Profiles for {name=}'):
        for model, number_of_examples, extract_func in product(MODELS, EXAMPLES, EXTRACTORS):
            instance = Instance(
                model,
                number_of_examples,
                extract_func,
            )

            with timeblock(f'Processing {instance}') as timer:
                if (profile := run_query_for_instance(instance, query)) is None:
                    continue

                extracted_profile = ExtractedProfile(
                    profile=profile,
                    model=instance.model,
                    number_of_examples=instance.number_of_examples,
                    extraction_function=instance.extract.__qualname__,
                    extraction_time=timer.elapsed_time,
                )

                profiles[len(profiles) + 1] = extracted_profile
                log(extracted_profile, use_pprint=True, log_file_name=extracted_profile_log)

    log('Creating tournament evaluation and ranking')
    with timeblock('Processing Evaluation'):
        tournament = tournament_ranking(EVALUATION_MODEL, query, profiles, do_shuffle=DO_SHUFFLE_DURING_EVALUATION)

    result = AuthorResult(
        tournament=tournament,
        profiles=profiles,
        titles=query.titles,
        author=query.author,
    )
    log(result, use_pprint=True, log_file_name=extracted_profile_log, level=LogLevel.DEBUG)
    return result


def run_query_for_instance(instance: Instance, query: Query) -> Profile | None:
    log(f'Running query for instance: {instance}')

    retriever_getter = get_retriever_getter(instance.number_of_examples)

    llm = OpenAILanguageModel(instance.model, debug_context_name=instance.extract.__qualname__)

    try:
        return instance.extract(query, retriever_getter, llm)
    except Exception as e:
        # print traceback to console and log to file
        import traceback

        traceback.print_exc()

        log(f'Error processing {instance=}', e, level=LogLevel.WARNING)
        log(f'Error processing {instance=}', e, log_file_name='logs/extraction_errors.log')

        # if e is keyboard interrupt, exit the program
        if isinstance(e, KeyboardInterrupt):
            raise e

        return None


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 2:
        sys.exit('Usage: python -m src <command> <query>')

    if sys.argv[1] == 'gen_example':
        from src.generate_references import generate_example_references

        generate_example_references(int(sys.argv[2]), REFERENCE_GENERATION_MODEL)

    if sys.argv[1] == 'gen_combination':
        from src.generate_references import generate_combination_references

        generate_combination_references(int(sys.argv[2]), REFERENCE_GENERATION_MODEL)

    if sys.argv[1] == 'gen_ranking':
        from src.generate_references import generate_ranking_references

        generate_ranking_references(int(sys.argv[2]), EVALUATION_MODEL, OTHER_REFERENCE_GENERATION_MODEL)

    if sys.argv[1] == 'author':
        # TODO replace with get_authors_of_kit(5) to get the top 5 authors if the author is 'all'
        authors_list = ['Sanders', 'Oberweis', 'Stiefelhagen'] if sys.argv[2] == 'all' else [sys.argv[2]]
        for author in authors_list:
            result = process_author(author, number_of_papers=5)
            log('-' * 50)
            dump_author_result_to_json(result)
            generate_html_file_for_tournament_evaluation(result)
            generate_html_file_for_tournament_ranking_result(result)
