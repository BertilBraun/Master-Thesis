from src.logic.database import add_element_to_database, database_size
from src.logic.types import Combination, Competency, Profile, Example


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
