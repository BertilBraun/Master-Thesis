import re
import random

import numpy as np


from src.cas_preprocess import CAS_KOMPETENZEN_CLEAN_PATH, CASSample
from src.log import log
from src.database import get_example_messages, get_retriever_getter
from src.language_model import OpenAILanguageModel, trim_text_to_token_length
from src.util import chunked_iterate, load_json, log_all_exceptions
from src.extraction_custom import extract_from_abstracts_custom, combine_profiles_into_one
from src.types import Example, Query, SystemMessage, AIMessage, HumanMessage, Profile, RetrieverGetter, LanguageModel


def extract_profiles_from_full_texts_custom_for_cas(
    query: Query, retriever: RetrieverGetter, llm: LanguageModel
) -> list[Profile]:
    prompts = [
        [
            SystemMessage(
                content="""You are a helpful research assistant tasked with identifying and cataloging professional competencies of a company from their yearly financial statement.

WARNING: This task is only possible if the company describes their general work, products, or services in the document. If the document is solely financial data, please inform the user that the task cannot be completed by responding with "NO DATA AVAILABLE."

Extract all relevant competencies that can be associated with the company from the text, and organize them into a structured competency profile as follows:
```
Domain: "[Short Domain Description]"
Competencies:
- [Competency Name]: [Detailed explanation of how Competency 1 is demonstrated in the text]
- [Competency Name]: [Detailed explanation of how Competency 2 is demonstrated in the text]
...
```
The domain should succinctly summarize the general area in which the company works. The competencies should be specific skills or knowledge areas demonstrated by the company. Ensure your analysis is neutral and precise, based solely on the content provided."""
            ),
            *get_example_messages(full_text, retriever(Example)),
            HumanMessage(
                content=f"""Please extract the professional competencies from this complete document text:
                
{trim_text_to_token_length(full_text, 6000)}


The domain should succinctly summarize the general area of work. The competencies should be specific skills or knowledge areas demonstrated by the company. This is the format:
```
Domain: "[Short Domain Description]"
Competencies:
- [Competency Name]: [Detailed explanation of how Competency 1 is demonstrated in the text]
- [Competency Name]: [Detailed explanation of how Competency 2 is demonstrated in the text]
...
```
Ensure your analysis is neutral and precise, based solely on the content of the paper provided.

WARNING: This task is only possible if the company describes their general work, products, or services in the document. If only information about the financial statement of the company is provided, please inform the user that the task cannot be completed by responding with "NO DATA AVAILABLE.". Financial statement preparation, accounting, asset / inventory etc. management and governance are NOT considered competencies for this task. I repeat: Under no circumstances should you include financial data or the management of financial data in the competency profile. Respond with "NO DATA AVAILABLE." if the document is solely financial data/description."""
            ),
        ]
        for full_text in query.full_texts
    ]

    llm_profiles = llm.batch(prompts, stop=['\n\n\n\n'])

    # Assuming conversion of profiles to string format and joining them happens here.
    profiles: list[Profile] = []
    for profile_str in llm_profiles:
        with log_all_exceptions('Error parsing profile in _extract_from_full_texts_custom'):
            if 'NO DATA AVAILABLE' not in profile_str:
                profiles.append(Profile.parse(profile_str))

    return profiles


def extract_profiles_from_query(query: Query, retriever: RetrieverGetter, llm: LanguageModel) -> list[Profile]:
    abstract_extractions = [
        extract_from_abstracts_custom(
            Query(
                full_texts=[],
                abstracts=[elem[0] for elem in elements],
                titles=[elem[1] for elem in elements],
                author=query.author,
            ),
            retriever,
            llm,
        )
        for elements in chunked_iterate(list(zip(query.abstracts, query.titles)), 5)
    ]

    fulltext_extractions = extract_profiles_from_full_texts_custom_for_cas(query, retriever, llm)

    return fulltext_extractions + abstract_extractions


def merge_profiles_recursively_into_one(
    profiles: list[Profile],
    retriever: RetrieverGetter,
    llm: LanguageModel,
    profiles_to_combine_at_a_time: int = 5,
) -> Profile:
    assert len(profiles) > 0, ''

    # merge the profiles recursively 5 at a time
    while len(profiles) > 1:
        profiles_to_combine = profiles[:profiles_to_combine_at_a_time]
        profiles = profiles[profiles_to_combine_at_a_time:]
        profiles.append(combine_profiles_into_one(profiles_to_combine, retriever, llm))

    return profiles[0]


def evaluate_profile_score_based_with_reference_description(
    profile: Profile, reference_description: str, llm: LanguageModel
) -> float:
    # compare the profile with the reference description
    # return a score between 1 and 100

    prompt = [
        SystemMessage(
            content="""You are an expert evaluator assigned to assess the relevance of a company's competency profile against a reference description. Each competency profile outlines the key domains of expertise and competencies of a company, particularly focusing on industries, products, and services offered. Your task is to evaluate how well a given company's profile aligns with the provided reference description.

You should carefully compare the competencies and industry focus listed in the company's profile with the branches, products, and areas of expertise mentioned in the reference description. Your evaluation should be neutral, accurate, and detailed.

Based on your assessment, provide a score between 1 and 100 that reflects the degree of correlation between the company's profile and the reference description. Structure your response using the following format:
```
Reasoning: [Your detailed analysis and reasoning]
Evaluation Score (0-100): [Your Evaluation Score]
```"""
        ),
        HumanMessage(
            content="""Please assess the following company profile in terms of its relevance to the provided reference description.

Profile:
Domain: "Expert in developing advanced manufacturing systems and tooling solutions for the metalworking industry."

Competencies:
- Precision Machining: Provides state-of-the-art machines for precision metalworking.
- Advanced Manufacturing Systems: Develops integrated systems for efficient and flexible production.
- Tooling Solutions: Offers comprehensive tooling systems tailored to metalworking needs.
- Customer Support: Delivers robust after-sales services and support.
- Innovation in Machining: Pioneers in the design and implementation of new machining technologies.

Reference:
Branches: Metallbearbeitungsmaschinen | Herstellung von Werkzeugmaschinen für die Metallbearbeitung | Werkzeugmaschinen | Herstellung von Werkzeugmaschinen für die Metallbearbeitung | Herstellung von Werkzeugmaschinen für die Metallbearbeitung | Werkzeugmaschinen

Description: HELLER wurde im Jahr 1894 in Nürtingen als kleiner Handwerksbetrieb gegründet. Heute entwickelt und produziert die Unternehmensgruppe als einer der führenden Hersteller weltweit modernste Werkzeugmaschinen und komplette Fertigungssysteme für die spanende Bearbeitung. Das HELLER Produktprogramm umfasst 4- und 5-achsige Bearbeitungszentren, Fräsdreh-Zentren, Maschinen für die Kurbel- und Nockenwellenbearbeitung, flexible Fertigungssysteme sowie ein modulares Dienstleistungsangebot., Entwicklung, Herstellung und Vertrieb von Maschinen, insbesondere von Werkzeugmaschinen, und Steuerungssystemen."""
        ),
        AIMessage(
            content="""Reasoning: The company profile shows a strong alignment with the reference description, particularly in areas of precision machining and advanced manufacturing systems. The focus on tooling solutions and customer support further enhances the relevance to the reference, although the reference description emphasizes specific types of machining centers (e.g., 4- and 5-axis centers) that are not explicitly mentioned in the profile.
Evaluation Score (0-100): 90"""
        ),
        HumanMessage(
            content=f"""Please assess the following competency profile in terms of its relevance to the provided reference description.

Profile:
{profile}


Reference:
{reference_description}


Your evaluation must follow this format:
```
Reasoning: [Your Evaluation and Reasoning]
Evaluation Score (0-100): [Your Evaluation Score]
```
Be specific and detailed in your reasoning and provide the score of the correlation to the reference."""
        ),
    ]
    response = llm.invoke(prompt)

    log('Scoring response:', response)

    # Use regex to find all floats in the response
    matches = re.findall(r'\d+\.\d+|\d+', response)

    # Convert the last match to a float and return it
    if matches:
        return float(matches[-1])
    else:
        raise ValueError('No evaluation score found in the response.')


def evaluate_sample(query: Query, reference_description: str) -> float:
    # extract profiles from the documents 5 docs at a time with extract from abstracts
    # extract 1 profile each from the jahresabschlüsse as extract from full texts
    # merge them iteratively with the combine part of the full texts extraction

    llm = OpenAILanguageModel('alias-large-instruct', 'cas_evaluation_documents')
    retriever = get_retriever_getter(1)

    base_profiles = extract_profiles_from_query(query, retriever, llm)
    random.shuffle(base_profiles)
    combined_profile = merge_profiles_recursively_into_one(base_profiles, retriever, llm)

    llm.debug_context_name = 'cas_evaluation_score'
    score = evaluate_profile_score_based_with_reference_description(combined_profile, reference_description, llm)

    log(
        'Name:',
        query.author,
        'profile:\n',
        str(combined_profile),
        '\n\n\n\n',
        'Reference:',
        reference_description,
        '\n\n\n',
        'Profile score:',
        score,
        'With given:',
        len(query.abstracts),
        'abstracts and',
        len(query.full_texts),
        'jahresabschlüsse',
        '\n\n\n\n',
    )

    return score


if __name__ == '__main__':
    dataset = load_json(CAS_KOMPETENZEN_CLEAN_PATH, CASSample)

    scores: list[float] = []

    for i, sample in enumerate(dataset):
        print(f'Processing sample {i + 1}/{len(dataset)} ({sample.name})')
        with log_all_exceptions(f'Error evaluating sample {sample.name}'):
            scores.append(
                evaluate_sample(
                    Query(
                        full_texts=sample.jahresabschlüsse,
                        abstracts=[f'Title: {doc.title}\n\nAbstract: {doc.abstract}' for doc in sample.documents],
                        titles=[doc.title for doc in sample.documents],
                        author=sample.name,
                    ),
                    f"Branches: {sample.reference.branches}\n\nDescriptions: {', '.join(sample.reference.descriptions)}",
                )
            )

    print(f'Scores: {scores}')
    print(f'Average score: {np.mean(scores)}')
    print(f'Min score: {np.min(scores)}')
    print(f'Max score: {np.max(scores)}')
    print(f'Std score: {np.std(scores)}')
    print(f'Variance score: {np.var(scores)}')
    print(f'Number of samples: {len(scores)}')
