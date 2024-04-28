import src.openai_defines  # noqa # sets the OpenAI API key and base URL to the environment variables

import time
import random

from tqdm import tqdm
from itertools import product

from src.language_model import OpenAILanguageModel
from src.evaluation import evaluate_with, tournament_ranking
from src.extraction_custom import (
    _extract_from_full_texts_custom,
    extract_from_abstracts_custom,
    extract_from_full_texts_custom,
    extract_from_summaries_custom,
)
from src.extraction_json import (
    extract_from_abstracts_json,
    extract_from_full_texts_json,
    extract_from_summaries_json,
)
from src.database import (
    add_element_to_database,
    database_size,
    get_retriever_getter,
    get_sample_from_database,
)
from src.display import generate_html_file_for_extraction_result
from src.papers import (
    get_authors_of_kit,
    get_papers_by_author,
    get_random_papers,
)
from src.types import (
    AuthorResult,
    Combination,
    Competency,
    DatabaseTypes,
    Evaluation,
    ExtractedProfile,
    Profile,
    Example,
    Instance,
    Query,
    Ranking,
)
from src.util import timeit
from src.log import LogLevel, datetime_str, log


OTHER_REFERENCE_GENERATION_MODEL = 'mistral'  # should be something other than the REFERENCE_GENERATION_MODEL to generate different results which can be used for ranking
REFERENCE_GENERATION_MODEL = 'neural'  # TODO should be something stronger like 'gpt-4-turbo'
EVALUATION_MODEL = 'neural'  # TODO should be something stronger like 'gpt-4-turbo'

MODELS = [
    'mistral',
    'neural',
    # 'mixtral',
    # TODO 'phi3 mini' 3.8B parameters
    # TODO 'llama3' 8B parameters (ultra sota)
    # Set src.openai_defines.BASE_URL_LLM = None for and set the API key and use one of the following models to run the inference on the OpenAI API
    # TODO 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'
]

EXAMPLES = [2, 1, 0]

EXTRACTORS = [
    extract_from_abstracts_custom,
    extract_from_abstracts_json,
    extract_from_summaries_custom,
    extract_from_summaries_json,
    extract_from_full_texts_custom,
    extract_from_full_texts_json,
][1::2]  # Only use the json extractors for now, as they are more reliable


@timeit('Processing Author')
def process_author(name: str, number_of_papers: int = 5) -> AuthorResult:
    profiles: list[ExtractedProfile] = []

    query = get_papers_by_author(name, number_of_papers=number_of_papers)

    extracted_profile_log = f'logs/extracted_profiles/{name}_{datetime_str()}.log'

    for model, number_of_examples, extract_func in tqdm(
        product(MODELS, EXAMPLES, EXTRACTORS),
        desc='Processing different models and extractors',
    ):
        instance = Instance(
            model,
            number_of_examples,
            extract_func,
        )

        start = time.time()
        if (profile := run_query_for_instance(instance, query)) is None:
            continue

        profiles.append(
            ExtractedProfile(
                profile=profile,
                model=instance.model,
                number_of_examples=instance.number_of_examples,
                extraction_function=instance.extract.__qualname__,
                extraction_time=time.time() - start,
            )
        )
        log(profiles[-1], use_pprint=True, log_file_name=extracted_profile_log)

    evaluation_result = evaluate_with(EVALUATION_MODEL, query, profiles)
    root, preferences = tournament_ranking(EVALUATION_MODEL, query, profiles)

    result = AuthorResult(
        evaluation_result=evaluation_result,
        root=root,
        preferences=preferences,
        titles=query.titles,
        author=query.author,
    )
    log(result, use_pprint=True, log_file_name=extracted_profile_log)
    return result


@timeit('Querying Instance')
def run_query_for_instance(instance: Instance, query: Query) -> Profile | None:
    log(f'Running query for instance: {instance}', level=LogLevel.INFO)

    retriever_getter = get_retriever_getter(instance.number_of_examples)

    llm = OpenAILanguageModel(instance.model, debug_context_name=instance.extract.__qualname__)

    try:
        profile = instance.extract(query, retriever_getter, llm)
    except Exception as e:
        log(f'Error processing {instance=}', e, level=LogLevel.WARNING)
        log(f'Error processing {instance=}', e, log_file_name='logs/extraction_errors.log')

        # if e is keyboard interrupt, exit the program
        if isinstance(e, KeyboardInterrupt):
            raise e

        return None

    return profile


def write_reference(element: DatabaseTypes, file_name: str) -> None:
    add_element_to_database(element, is_reference=True)

    log(element, use_pprint=True, log_file_name=file_name)
    log('\n\n\n\n\n', log_file_name=file_name)


def generate_example_references(number_of_references_to_generate: int):
    add_initial_example_references()

    # Use the actual OpenAI API not the LocalAI for generating as best results are expected from the largest models
    # TODO src.openai_defines.BASE_URL_LLM = None

    # Get papers from different topics
    queries = get_random_papers(number_of_references_to_generate)

    generated_examples_file = f'logs/generated_example_references/{datetime_str()}.log'

    for query in queries:
        # Use one abstract at a time in a 1 shot prompt
        instance = Instance(
            REFERENCE_GENERATION_MODEL,
            number_of_examples=1,
            extract=extract_from_abstracts_json,
        )

        if (profile := run_query_for_instance(instance, query)) is None:
            continue

        example = Example(abstract=query.abstracts[0], profile=profile)

        # Write the extracted Profile as reference to a file and database
        write_reference(example, generated_examples_file)


def generate_combination_references(number_of_references_to_generate: int):
    add_initial_combination_references()

    # Use the actual OpenAI API not the LocalAI for generating as best results are expected from the largest models
    # TODO src.openai_defines.BASE_URL_LLM = None

    # Get a random subset of authors to generate the references from
    authors = get_authors_of_kit(number_of_references_to_generate * 2)
    random.shuffle(authors)
    authors = authors[:number_of_references_to_generate]

    # Get papers from different authors but only look at the abstracts
    queries = [get_papers_by_author(author.name, number_of_papers=3) for author in authors]
    queries = [
        Query(
            full_texts=query.abstracts,
            abstracts=query.abstracts,
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


def generate_evaluation_references(number_of_references_to_generate: int):
    add_initial_evaluation_references()

    # Use the actual OpenAI API not the LocalAI for generating as best results are expected from the largest models
    # TODO src.openai_defines.BASE_URL_LLM = None

    examples = get_sample_from_database(Example, number_of_references_to_generate)

    generated_evaluations_file = f'logs/generated_evaluation_references/{datetime_str()}.log'

    for example in examples:
        evaluation_result = evaluate_with(
            REFERENCE_GENERATION_MODEL,
            Query(
                full_texts=['Unknown'],
                abstracts=[example.abstract],
                titles=['Unknown'],
                author='Unknown',
            ),
            [ExtractedProfile.from_profile(example.profile)],
        )

        evaluation = Evaluation(
            paper_text=example.abstract,
            profile=example.profile,
            reasoning=evaluation_result[0].reasoning,
            score=evaluation_result[0].score,
        )

        # Write the evaluation as reference to a file and database
        write_reference(evaluation, generated_evaluations_file)


def generate_ranking_references(number_of_references_to_generate: int):
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
            Instance(model=OTHER_REFERENCE_GENERATION_MODEL, number_of_examples=0, extract=extract_from_abstracts_json),
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

        root, preferences = tournament_ranking(
            REFERENCE_GENERATION_MODEL,
            query,
            [
                ExtractedProfile.from_profile(example.profile),
                ExtractedProfile.from_profile(other_profile),
            ],
        )

        ranking = Ranking(
            paper_text=example.abstract,
            reasoning=root.match.reasoning,
            profiles=(root.match.profiles[0].profile, root.match.profiles[1].profile),
            preferred_profile=root.match.preferred_profile,
        )

        # Write the evaluation as reference to a file and database
        write_reference(ranking, generated_rankings_file)


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


def add_initial_combination_references():
    # Add some initial references to the database.
    if database_size(Combination) > 0:
        return  # Do not add references if there are already references in the database

    add_element_to_database(
        Combination(
            input_profiles=[
                Profile(
                    domain='Electrochemical and Cycling Characterization of Supercapacitors',
                    competencies=[
                        Competency(
                            name='Experimental Design',
                            description='The authors demonstrate competency in '
                            'experimental design by conducting '
                            'three-electrode cyclic voltammetry '
                            'experiments at various scan rates to '
                            'evaluate the specific capacitance of the '
                            'microporous activated carbon.',
                        ),
                        Competency(
                            name='Data Analysis',
                            description='The authors show competency in data analysis '
                            'by measuring the specific capacitance of the '
                            'microporous activated carbon and determining '
                            'the maximum operating potential range.',
                        ),
                        Competency(
                            name='Material Characterization',
                            description='The authors exhibit competency in material '
                            'characterization by assessing the '
                            'performance of the microporous activated '
                            'carbon in an ionic liquid electrolyte at '
                            'different temperatures.',
                        ),
                        Competency(
                            name='Electrolyte Evaluation',
                            description='The authors demonstrate competency in '
                            'evaluating electrolytes by using '
                            'N-butyl-N-methylpyrrolidinium '
                            'bis(trifluoromethanesulfonyl)imide '
                            '(PYR14TFSI) ionic liquid as the electrolyte '
                            'and studying its impact on the '
                            "supercapacitor's performance.",
                        ),
                        Competency(
                            name='Temperature Management',
                            description='The authors show competency in temperature '
                            'management by conducting experiments at '
                            'various temperatures and identifying the '
                            'suitability of the supercapacitor for '
                            'high-temperature applications (≥60°C).',
                        ),
                        Competency(
                            name='Cycling Stability',
                            description='The authors demonstrate competency in '
                            'assessing cycling stability by cycling the '
                            'coin cell for 40,000 cycles without any '
                            'change in cell resistance and evaluating the '
                            'high stable specific capacitance in the '
                            'ionic liquid electrolyte.',
                        ),
                    ],
                ),
                Profile(
                    domain='Energy Storage Systems and Battery Technology',
                    competencies=[
                        Competency(
                            name='Scientific Literature Review',
                            description='The author demonstrates the ability to '
                            'conduct a comprehensive review of existing '
                            'literature on the lithium/air battery '
                            'system, as evidenced by the discussion of '
                            'various studies and research efforts by top '
                            'academic and industrial laboratories '
                            'worldwide.',
                        ),
                        Competency(
                            name='Critical Evaluation',
                            description='The author showcases the skill to critically '
                            'evaluate the progress made in the '
                            'development of the Li/air electrochemical '
                            'system, highlighting the issues that have '
                            'been identified and the breakthroughs '
                            'achieved.',
                        ),
                        Competency(
                            name='Forecasting and Trend Analysis',
                            description='The competency to forecast and analyze '
                            'future R&D trends in the battery technology '
                            "field is demonstrated by the author's "
                            'attempt to propose potential future '
                            'directions for research in the Li/air '
                            'system.',
                        ),
                        Competency(
                            name='Communication and Synthesis',
                            description='The author effectively communicates complex '
                            'scientific concepts in a clear and concise '
                            'manner, synthesizing information from '
                            'various sources to provide a comprehensive '
                            "overview of the lithium/air battery system's "
                            'current state and potential future impact.',
                        ),
                    ],
                ),
                Profile(
                    domain='Aqueous Rechargeable Batteries and Renewable Energy',
                    competencies=[
                        Competency(
                            name='Research and Development',
                            description='The paper highlights the importance of '
                            'aqueous rechargeable batteries in the '
                            'development of renewable energy sources, '
                            'indicating a deep understanding of current '
                            'energy demands and the role of '
                            'cost-efficiency in battery technology.',
                        ),
                        Competency(
                            name='Electrode Materials',
                            description='The author demonstrates knowledge of '
                            'electrode materials and their improvement '
                            'over the past decade, which contributes to '
                            'the efficiency of aqueous battery systems.',
                        ),
                        Competency(
                            name='Electrolytes',
                            description='The paper emphasizes the use of highly '
                            'concentrated aqueous electrolytes in battery '
                            'systems and their impact on energy density, '
                            'cyclability, and safety.',
                        ),
                        Competency(
                            name='Strategic Innovation',
                            description='The author provides a summary of the '
                            'strategies proposed to overcome the hurdles '
                            'limiting present aqueous battery '
                            'technologies, showcasing an ability to '
                            'identify and innovate to address challenges '
                            'in the field.',
                        ),
                        Competency(
                            name='Focused Specialization',
                            description='The paper focuses on aqueous batteries for '
                            'lithium and post-lithium chemistries, '
                            'demonstrating a specialized understanding of '
                            'these specific battery types and their '
                            'potential for improved energy density.',
                        ),
                        Competency(
                            name='Synthesis and Analysis',
                            description='The author synthesizes and analyzes the '
                            'unique advantages of concentrated '
                            'electrolytes in aqueous battery systems, '
                            'contributing to a comprehensive '
                            'understanding of the subject matter.',
                        ),
                        Competency(
                            name='Timely Information Dissemination',
                            description='The Review aims to provide a timely summary '
                            'of the advances in aqueous battery systems, '
                            'indicating an awareness of the need for '
                            'current and relevant information in the '
                            'field.',
                        ),
                    ],
                ),
            ],
            combined_profile=Profile(
                domain='Advanced Energy Storage Technologies',
                competencies=[
                    Competency(
                        name='Material Characterization',
                        description='The assessment of microporous activated '
                        'carbon performance in ionic liquid '
                        'electrolyte at different temperatures '
                        "highlights the authors' competency in "
                        'material characterization.',
                    ),
                    Competency(
                        name='Electrolyte Evaluation',
                        description='The use of N-butyl-N-methylpyrrolidinium '
                        'bis(trifluoromethanesulfonyl)imide '
                        '(PYR14TFSI) ionic liquid as the electrolyte '
                        'and its impact on supercapacitor '
                        'performance showcases competency in '
                        'evaluating electrolytes.',
                    ),
                    Competency(
                        name='Temperature Management',
                        description='Conducting experiments at various '
                        'temperatures and identifying suitability '
                        'for high-temperature applications (≥60°C) '
                        'demonstrates competency in temperature '
                        'management.',
                    ),
                    Competency(
                        name='Forecasting and Trend Analysis',
                        description='The proposal of potential future directions '
                        'for research in the Li/air system indicates '
                        'competency in forecasting and trend '
                        'analysis.',
                    ),
                    Competency(
                        name='Strategic Innovation',
                        description='Identifying and proposing strategies to '
                        'overcome hurdles limiting present aqueous '
                        'battery technologies showcases competency '
                        'in strategic innovation.',
                    ),
                    Competency(
                        name='Synthesis and Analysis',
                        description='The synthesis and analysis of concentrated '
                        "electrolytes' unique advantages in aqueous "
                        'battery systems contribute to overall '
                        'competency.',
                    ),
                ],
            ),
        ),
        is_reference=True,
    )
    # add_element_to_database(Combination(), is_reference=True)


def add_initial_evaluation_references():
    # Add some initial references to the database.
    if database_size(Evaluation) > 0:
        return  # Do not add references if there are already references in the database

    add_element_to_database(
        Evaluation(
            paper_text='\n'
            'A novel class of metal organic frameworks (MOFs) has been synthesized from Cu-acetate and '
            'dicarboxylic acids using liquid phase epitaxy. The SURMOF-2 isoreticular series exhibits P4 '
            'symmetry, for the longest linker a channel-size of 3 × 3 nm2 is obtained, one of the largest '
            'values reported for any MOF so far. High quality, ab-initio electronic structure calculations '
            'confirm the stability of a regular packing of (Cu++)2- carboxylate paddle-wheel planes with P4 '
            'symmetry and reveal, that the SURMOF-2 structures are in fact metastable, with a fairly large '
            'activation barrier for the transition to the bulk MOF-2 structures exhibiting a lower, twofold '
            '(P2 or C2) symmetry. The theoretical calculations also allow identifying the mechanism for the '
            'low-temperature epitaxial growth process and to explain, why a synthesis of this highly '
            'interesting, new class of high-symmetry, metastable MOFs is not possible using the conventional '
            'solvothermal process.',
            profile=Profile(
                domain='Materials Science and Chemistry',
                competencies=[
                    Competency(
                        name='Synthesis Expertise',
                        description='The ability to synthesize new materials using '
                        'innovative methods, as demonstrated by the creation '
                        'of a novel class of metal organic frameworks (MOFs) '
                        'from Cu-acetate and dicarboxylic acids using liquid '
                        'phase epitaxy.',
                    ),
                    Competency(
                        name='Structural Analysis',
                        description='Proficiency in analyzing the structure of complex '
                        'materials, as evidenced by the identification of the '
                        'P4 symmetry and channel-size of 3 × 3 nm2 in the '
                        'SURMOF-2 isoreticular series.',
                    ),
                    Competency(
                        name='Electronic Structure Calculations',
                        description='Skill in performing high-quality, ab-initio '
                        'electronic structure calculations to confirm the '
                        'stability of material structures. This is '
                        'demonstrated by the confirmation of regular packing '
                        'of (Cu++)2- carboxylate paddle-wheel planes with P4 '
                        'symmetry in SURMOF-2 structures.',
                    ),
                    Competency(
                        name='Stability Assessment',
                        description='Expertise in assessing the stability and '
                        'metastability of materials. The text showcases this '
                        'competency by revealing that SURMOF-2 structures are '
                        'metastable with a fairly large activation barrier for '
                        'the transition to bulk MOF-2 structures exhibiting '
                        'lower symmetry.',
                    ),
                    Competency(
                        name='Mechanism Identification',
                        description='Proficiency in identifying the mechanisms behind '
                        'material synthesis processes, as illustrated by the '
                        'identification of the mechanism for the '
                        'low-temperature epitaxial growth process of the novel '
                        'MOFs.',
                    ),
                    Competency(
                        name='Problem Solving',
                        description='Ability to address and explain complex scientific '
                        'challenges, as demonstrated by the explanation of why '
                        'a synthesis of the highly interesting, new class of '
                        'high-symmetry, metastable MOFs is not possible using '
                        'the conventional solvothermal process.',
                    ),
                ],
            ),
            reasoning="""The competency profile aligns well with the themes and expertise areas presented in the scientific abstracts. The abstracts focus on the synthesis, structural analysis, electronic structure calculations, stability assessment, mechanism identification, and problem-solving aspects of a novel class of metal organic frameworks (MOFs). The profile competencies match these themes closely, as demonstrated by specific examples from the abstracts:

1. Synthesis Expertise: The synthesis of the novel MOFs using liquid phase epitaxy is a clear example of this competency (Abstract).
2. Structural Analysis: The identification of P4 symmetry and channel-size in the SURMOF-2 isoreticular series showcases structural analysis expertise (Abstract).
3. Electronic Structure Calculations: The use of high-quality, ab-initio electronic structure calculations to confirm the stability of the MOF structures demonstrates this competency (Abstract).
4. Stability Assessment: The revelation of the metastable nature of SURMOF-2 structures and the activation barrier for transition to bulk MOF-2 structures highlights stability assessment competency (Abstract).
5. Mechanism Identification: The identification of the low-temperature epitaxial growth process mechanism is an example of mechanism identification competency (Abstract).        
6. Problem Solving: The explanation of why the synthesis of these MOFs is not possible using conventional solvothermal processes demonstrates problem-solving ability (Abstract). 

Overall, the profile's domain of "Materials Science and Chemistry" closely aligns with the focus areas of the abstracts, further supporting the relevance of the competencies listed in the profile.""",
            score=90,
        ),
        is_reference=True,
    )

    add_element_to_database(
        Evaluation(
            paper_text='\n'
            'A novel class of metal organic frameworks (MOFs) has been synthesized from Cu-acetate and '
            'dicarboxylic acids using liquid phase epitaxy. The SURMOF-2 isoreticular series exhibits P4 '
            'symmetry, for the longest linker a channel-size of 3 × 3 nm2 is obtained, one of the largest '
            'values reported for any MOF so far. High quality, ab-initio electronic structure calculations '
            'confirm the stability of a regular packing of (Cu++)2- carboxylate paddle-wheel planes with P4 '
            'symmetry and reveal, that the SURMOF-2 structures are in fact metastable, with a fairly large '
            'activation barrier for the transition to the bulk MOF-2 structures exhibiting a lower, twofold '
            '(P2 or C2) symmetry. The theoretical calculations also allow identifying the mechanism for the '
            'low-temperature epitaxial growth process and to explain, why a synthesis of this highly '
            'interesting, new class of high-symmetry, metastable MOFs is not possible using the conventional '
            'solvothermal process.',
            profile=Profile(
                domain='Materials Science and Chemistry',
                competencies=[
                    Competency(
                        name='Synthesis Expertise',
                        description='The ability to synthesize new materials using '
                        'innovative methods, as demonstrated by the creation '
                        'of a novel class of metal organic frameworks (MOFs) '
                        'from Cu-acetate and dicarboxylic acids using liquid '
                        'phase epitaxy.',
                    ),
                    Competency(
                        name='Structural Analysis',
                        description='Proficiency in analyzing the structure of complex '
                        'materials, as evidenced by the identification of the '
                        'P4 symmetry and channel-size of 3 × 3 nm2 in the '
                        'SURMOF-2 isoreticular series.',
                    ),
                    Competency(
                        name='Electronic Structure Calculations',
                        description='Skill in performing high-quality, ab-initio '
                        'electronic structure calculations to confirm the '
                        'stability of material structures. This is '
                        'demonstrated by the confirmation of regular packing '
                        'of (Cu++)2- carboxylate paddle-wheel planes with P4 '
                        'symmetry in SURMOF-2 structures.',
                    ),
                    Competency(
                        name='Stability Assessment',
                        description='Expertise in assessing the stability and '
                        'metastability of materials. The text showcases this '
                        'competency by revealing that SURMOF-2 structures are '
                        'metastable with a fairly large activation barrier for '
                        'the transition to bulk MOF-2 structures exhibiting '
                        'lower symmetry.',
                    ),
                    Competency(
                        name='Mechanism Identification',
                        description='Proficiency in identifying the mechanisms behind '
                        'material synthesis processes, as illustrated by the '
                        'identification of the mechanism for the '
                        'low-temperature epitaxial growth process of the novel '
                        'MOFs.',
                    ),
                    Competency(
                        name='Problem Solving',
                        description='Ability to address and explain complex scientific '
                        'challenges, as demonstrated by the explanation of why '
                        'a synthesis of the highly interesting, new class of '
                        'high-symmetry, metastable MOFs is not possible using '
                        'the conventional solvothermal process.',
                    ),
                ],
            ),
            reasoning='The competency profile aligns closely with the themes and expertise areas presented in the '
            'abstracts. The abstracts discuss the synthesis of a novel class of MOFs using liquid phase '
            'epitaxy, which is directly related to the Synthesis Expertise competency in the profile. The '
            'structural analysis of the SURMOF-2 isoreticular series, including the identification of P4 '
            'symmetry and channel size, corresponds well with the Structural Analysis competency. The '
            'electronic structure calculations performed to confirm the stability of the material structures '
            'align with the Electronic Structure Calculations competency.\n'
            '\n'
            'The profile also highlights Stability Assessment, Mechanism Identification, and Problem Solving '
            'competencies, which are all reflected in the abstracts. The text reveals the metastability of '
            "SURMOF-2 structures, demonstrating the expert's ability to assess stability and metastability. "
            'The identification of the mechanism for the low-temperature epitaxial growth process and the '
            'explanation of why the synthesis of the new MOFs cannot be achieved using conventional '
            'solvothermal processes showcase the Mechanism Identification and Problem Solving competencies, '
            'respectively.\n'
            '\n'
            'Overall, the competency profile is highly coherent with the focus areas of the abstracts, with '
            'specific competencies being well-represented and no significant gaps identified.',
            score=95,
        ),
        is_reference=True,
    )


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 2:
        sys.exit('Usage: python -m src <command> <query>')

    if sys.argv[1] == 'add' and sys.argv[2] == 'references':
        add_initial_example_references()

    if sys.argv[1] == 'gen_example':
        generate_example_references(int(sys.argv[2]))

    if sys.argv[1] == 'gen_combination':
        generate_combination_references(int(sys.argv[2]))

    if sys.argv[1] == 'gen_evaluation':
        generate_evaluation_references(int(sys.argv[2]))

    if sys.argv[1] == 'gen_ranking':
        generate_ranking_references(int(sys.argv[2]))

    if sys.argv[1] == 'author':
        result = process_author(sys.argv[2], number_of_papers=4)

        log('Final result:')
        log(result, use_pprint=True)
        log('-' * 50)
        generate_html_file_for_extraction_result(result)
