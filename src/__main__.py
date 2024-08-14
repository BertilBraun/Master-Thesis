from itertools import product
import os
from time import sleep
import urllib.parse

from src.defines import OPENAI_API_KEY
from src.language_model import OpenAILanguageModel
from src.evaluation import pseudo_tournament_ranking, tournament_ranking
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
from src.papers import get_papers_by_author_cached
from src.types import (
    AuthorResult,
    ExtractedProfile,
    Profile,
    Instance,
    Query,
)
from src.util import load_json, log_all_exceptions, timeblock, timeit
from src.log import LogLevel, datetime_str, log


OTHER_REFERENCE_GENERATION_MODEL = 'mistral'  # should be something other than the REFERENCE_GENERATION_MODEL to generate different results which can be used for ranking
REFERENCE_GENERATION_MODEL = 'neural'  # TODO should be something stronger like 'gpt-4-turbo'
EVALUATION_MODEL = 'neural'  # TODO should be something stronger like 'gpt-4-turbo'

OTHER_REFERENCE_GENERATION_MODEL = 'alias-fast-instruct'
REFERENCE_GENERATION_MODEL = 'alias-fast-instruct'
EVALUATION_MODEL = 'alias-large-instruct'

DO_SHUFFLE_DURING_EVALUATION = True
RUN_EVALUATION = False

MODELS = [
    'dev-phi-3-mini',  # 3.8B parameters
    # TODO reactivate 'dev-phi-3-medium',  # 14B parameters
    'alias-large-instruct',  # mixtral 8x7B parameters
    'alias-fast-instruct',  # TODO deactivate
    # 'dev-llama-3-large',  # 70B parameters
    # 'dev-llama-3-small',  # 8B parameters
    # 'dev-gemma-large',  # 27B parameters
    # 'dev-gemma-small',  # 9B parameters
    # Set src.openai_defines.BASE_URL_LLM = None for and set the API key and use one of the following models to run the inference on the OpenAI API
    # TODO 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'
]
GPT_MODEL_TO_USE = 'gpt-4o-mini'


EXAMPLES = [1, 0][:1]  # Only use 1 example for now, as it seems to return better results

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

    query = get_papers_by_author_cached(name, number_of_papers=number_of_papers, KIT_only=True)

    extracted_profile_log = f'logs/extracted_profiles/{name}_{datetime_str()}.log'

    with timeblock(f'Processing Extracted Profiles for {name=}'):
        for model in MODELS:
            for number_of_examples, extract_func in product(EXAMPLES, EXTRACTORS):
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
            sleep(120)
    print(f'Actual timing was: {120*len(MODELS)}seconds faster than logged above')

    # Load profile from finetuned model based on author name
    if os.path.exists(f'finetuned_profile_{name}.json'):
        finetuned_profile = ExtractedProfile.from_json(load_json(f'finetuned_profile_{name}.json'))
    else:
        finetuned_profile = ExtractedProfile.from_profile(Profile('None Existant Finetuning Extraction', []))

    profiles[len(profiles) + 1] = finetuned_profile

    # Extract profiles via GPT4o-mini
    for number_of_examples, extract_func in product(EXAMPLES, EXTRACTORS):
        with timeblock(f'Processing GPT4o for {extract_func.__qualname__} ({number_of_examples})') as timer:
            retriever_getter = get_retriever_getter(number_of_examples)

            llm = OpenAILanguageModel(
                GPT_MODEL_TO_USE,
                debug_context_name=extract_func.__qualname__,
                base_url=None,
                api_key=OPENAI_API_KEY,
            )

            with log_all_exceptions(f'Error processing GPT4o for {extract_func.__qualname__} ({number_of_examples})'):
                profile = extract_func(query, retriever_getter, llm)

                extracted_profile = ExtractedProfile(
                    profile=profile,
                    model=GPT_MODEL_TO_USE,
                    number_of_examples=number_of_examples,
                    extraction_function=extract_func.__qualname__,
                    extraction_time=timer.elapsed_time,
                )

                profiles[len(profiles) + 1] = extracted_profile
                log(extracted_profile, use_pprint=True, log_file_name=extracted_profile_log)

    log('Creating tournament evaluation and ranking')
    if RUN_EVALUATION:
        with timeblock('Processing Evaluation'):
            tournament = tournament_ranking(EVALUATION_MODEL, query, profiles, do_shuffle=DO_SHUFFLE_DURING_EVALUATION)
    else:
        tournament = pseudo_tournament_ranking(profiles, do_shuffle=DO_SHUFFLE_DURING_EVALUATION)

    result = AuthorResult(
        tournament=tournament,
        profiles=profiles,
        titles=query.titles,
        author=query.author,
    )
    log(result, use_pprint=True, log_file_name=extracted_profile_log, level=LogLevel.DEBUG)
    return result


@timeit('Processing Author list')
def process_author_list(names: list[str], number_of_papers: int = 5) -> list[AuthorResult]:
    log(f'Processing Authors: {names=} {number_of_papers=}')
    profiles: dict[str, dict[int, ExtractedProfile]] = {name: {} for name in names}

    queries = {
        name: get_papers_by_author_cached(name, number_of_papers=number_of_papers, KIT_only=True) for name in names
    }

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
            pass  # sleep(120)

    # Load profile from finetuned model based on author name
    for name in names:
        if os.path.exists(f'finetuned_profile_{name}.json'):
            finetuned_profile = ExtractedProfile.from_json(load_json(f'finetuned_profile_{name}.json'))
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


def generate_mail_for_author_result(result: AuthorResult) -> None:
    MAIL_TEMPLATE = """**Subject:** Request for Participation in Competency Profile Evaluation

---  
    
Dear [NAME],

I hope this mail finds you well. As part of my Master's thesis and the research project [Kompetenznetzwerk](https://www.for.kit.edu/kompetenznetzwerk.php) of the KIT, I am conducting a study to evaluate the accuracy of various methods for automatically extracting competency profiles from research papers. Your expertise and feedback would be immensely valuable to this research.

The evaluation involves comparing personalized competency profiles that have been generated based on your five most cited papers. Since these profiles are derived from your own work, your input is crucial in determining which profile best reflects your competencies. The process should take no more than five minutes of your time, and detailed instructions are provided directly on the webpage to guide you through each step.

To participate, simply click on the link below to access the evaluation page:

**[Start the Evaluation]([LINK])**

Your participation is greatly appreciated and will contribute significantly to the development of more accurate competency extraction methods.

Thank you very much for your time and assistance!

Best regards,  
Bertil Braun  

KIT - Karlsruhe Institute of Technology  
[bertil.braun@student.kit.edu](mailto:bertil.braun@student.kit.edu)  
+49 1525 3810140

---

Link to the evaluation page: [LINK]  
Link to the Research Project Kompetenznetzwerk: https://www.for.kit.edu/kompetenznetzwerk.php"""

    # Link will be https://evaluation.tiiny.site/Christof%20W%C3%B6ll.evaluation.html for example for Christof Wöll
    link = f'https://evaluation.tiiny.site/{result.author.replace(" ", "%20")}.evaluation.html'

    mail = MAIL_TEMPLATE.replace('[NAME]', result.author).replace('[LINK]', link)

    base_url = 'https://www.digitalocean.com/community/markdown#?md='
    encoded_content = urllib.parse.quote(mail)

    render_link = base_url + encoded_content

    with open(f'results/{result.author}.mail.txt', 'w') as f:
        f.write(f'Render link: {render_link}\n\n\n\n')
        f.write(mail)


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
        authors_list = ['Sanders', 'Oberweis', 'Stiefelhagen'] if sys.argv[2] == 'all' else [sys.argv[2]]
        for result in process_author_list(authors_list):
            log('-' * 50)
            dump_author_result_to_json(result)
            generate_html_file_for_tournament_evaluation(result)
            generate_html_file_for_tournament_ranking_result(result)
            generate_mail_for_author_result(result)
