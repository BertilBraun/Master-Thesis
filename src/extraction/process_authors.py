import os
from itertools import product
from time import sleep

from src.defines import OPENAI_API_KEY
from src.hyperparameter_defines import (
    EXAMPLES,
    MODELS,
    GPT_MODEL_TO_USE,
    RUN_EVALUATION,
    DO_SHUFFLE_DURING_EVALUATION,
    EXTRACTORS,
    EVALUATION_MODEL,
)
from src.logic.language_model import OpenAILanguageModel
from src.extraction.evaluation import pseudo_tournament_ranking, tournament_ranking

from src.logic.database import get_retriever_getter
from src.logic.papers import get_papers_by_author_cached
from src.logic.types import AuthorResult, ExtractedProfile, Profile, Instance, Query
from src.util import load_json, log_all_exceptions, timeblock, timeit
from src.util.log import LogLevel, log


def process_author_list(names: list[str], number_of_papers: int = 5) -> list[AuthorResult]:
    queries = {
        name: get_papers_by_author_cached(name, number_of_papers=number_of_papers, KIT_only=True) for name in names
    }

    return process_author_map(queries)


@timeit('Processing Author map')
def process_author_map(queries: dict[str, Query]) -> list[AuthorResult]:
    log(f'Processing Authors: {list(queries.keys())}')
    profiles: dict[str, dict[int, ExtractedProfile]] = {name: {} for name in queries.keys()}

    for model in MODELS:
        for name, query in queries.items():
            for number_of_examples, extract_func in product(EXAMPLES, EXTRACTORS):
                instance = Instance(model, number_of_examples, extract_func)

                with timeblock(f'Processing {instance} for {name}') as timer:
                    if (profile := run_query_for_instance(instance, query)) is None:
                        continue

                    extracted_profile = ExtractedProfile(
                        profile=profile,
                        model=instance.model,
                        number_of_examples=instance.number_of_examples,
                        extraction_function=instance.extract.__qualname__,
                        extraction_time=timer.elapsed_time,
                    )

                    profiles[name][len(profiles[name]) + 1] = extracted_profile

        if model != MODELS[-1]:
            sleep(120)

    # Load profile from finetuned model based on author name
    for name in queries.keys():
        if os.path.exists(f'evaluation/finetuned_profile_{name}.json'):
            finetuned_profile = ExtractedProfile.from_json(load_json(f'evaluation/finetuned_profile_{name}.json'))
        else:
            finetuned_profile = ExtractedProfile.from_profile(Profile('None Existant Finetuning Extraction', []))

        profiles[name][len(profiles[name]) + 1] = finetuned_profile

    # Extract profiles via GPT4o-mini
    for name, query in queries.items():
        for number_of_examples, extract_func in product(EXAMPLES, EXTRACTORS):
            desc = f'{GPT_MODEL_TO_USE} for {extract_func.__qualname__} ({number_of_examples})'
            with timeblock(f'Processing {desc}') as timer:
                llm = OpenAILanguageModel(
                    GPT_MODEL_TO_USE,
                    debug_context_name=extract_func.__qualname__,
                    base_url=None,
                    api_key=OPENAI_API_KEY,
                )

                with log_all_exceptions(f'Error processing {desc}'):
                    profile = extract_func(query, get_retriever_getter(number_of_examples), llm)

                    extracted_profile = ExtractedProfile(
                        profile=profile,
                        model=GPT_MODEL_TO_USE,
                        number_of_examples=number_of_examples,
                        extraction_function=extract_func.__qualname__,
                        extraction_time=timer.elapsed_time,
                    )

                    profiles[name][len(profiles[name]) + 1] = extracted_profile

    log('Creating tournament evaluation and ranking')
    results: list[AuthorResult] = []

    for name, query in queries.items():
        if RUN_EVALUATION:
            with timeblock('Processing Evaluation'):
                tournament = tournament_ranking(
                    EVALUATION_MODEL, query, profiles[name], do_shuffle=DO_SHUFFLE_DURING_EVALUATION
                )
        else:
            tournament = pseudo_tournament_ranking(profiles[name], do_shuffle=DO_SHUFFLE_DURING_EVALUATION)

        result = AuthorResult(
            tournament=tournament,
            profiles=profiles[name],
            titles=query.titles,
            author=query.author,
        )
        log(result, use_pprint=True, level=LogLevel.DEBUG)
        results.append(result)

    return results


def run_query_for_instance(instance: Instance, query: Query) -> Profile | None:
    retriever_getter = get_retriever_getter(instance.number_of_examples)

    llm = OpenAILanguageModel(instance.model, debug_context_name=instance.extract.__qualname__)

    with log_all_exceptions(f'Error processing {instance=}'):
        return instance.extract(query, retriever_getter, llm)
