from collections import Counter, defaultdict
from itertools import product
import random
import numpy as np
from typing import Callable, Literal, TypedDict
from scipy.stats import spearmanr, kendalltau
from tabulate import tabulate
from tqdm import tqdm

import src.defines
from src.logic.jsonbin import JsonBin
from src.logic.types.base_types import Profile
from src.logic.types import EvaluationResult, RankingResult
from src.logic.database import get_retriever_getter
from src.logic.openai_language_model import OpenAILanguageModel
from src.logic.types.database_types import EvaluationResult_from_invalid_response, Ranking
from src.logic.types.message_types import AIExampleMessage, HumanExampleMessage, HumanMessage, Message, SystemMessage
from src.extraction.evaluation import prompt_for_ranking
from src.util import log

random.seed(42)  # For reproducibility


class UploadedProfile(TypedDict):
    model_name: str
    profile: Profile


class UploadedProfiles(TypedDict):
    author: str
    profiles: list[UploadedProfile]
    abstracts: list[str]


def pprint(obj):
    from pprint import pprint as pp

    pp(obj, width=200)


def p_values_combined(p_values: list[float]) -> float:
    # Stouffer’s method for combining p-values
    # Tippett's method for combining p-values
    return (1 - np.prod(1 - np.array(p_values))).item()
    # Fisher's method for combining p-values
    return -2 * np.sum(np.log(p_values))


# Function to update Elo ratings based on the results
def get_elo_ratings(results: list[RankingResult], k: float = 32.0) -> dict[int, float]:
    elo_ratings: dict[int, float] = {}

    for result in results:
        profile1_index, profile2_index = result.profiles
        rating1 = elo_ratings.get(profile1_index, 1000.0)
        rating2 = elo_ratings.get(profile2_index, 1000.0)

        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        expected2 = 1 - expected1  # Since expected1 + expected2 = 1

        if result.preferred_profile_index == 1:
            score1, score2 = 1.0, 0.0
        elif result.preferred_profile_index == 2:
            score1, score2 = 0.0, 1.0
        else:  # Draw
            score1 = score2 = 0.5

        # Update ratings
        elo_ratings[profile1_index] = rating1 + k * (score1 - expected1)
        elo_ratings[profile2_index] = rating2 + k * (score2 - expected2)

    return elo_ratings


# Function to run pairwise evaluations
def run_pairwise_evaluations(
    profiles: list[UploadedProfile], evaluator: Callable[[Profile, Profile], list[EvaluationResult]]
) -> list[RankingResult]:
    results: list[RankingResult] = []

    for i, profile1 in enumerate(tqdm(profiles, desc='Evaluating profiles')):
        for j, profile2 in enumerate(profiles):
            if j <= i:
                continue

            # TODO maybe not all with all, but only i vs i+1 or sth like that for better results
            # if j != i + 1:
            #     continue

            evaluation = evaluator(profile1['profile'], profile2['profile'])

            for result in evaluation:
                results.append(
                    RankingResult(
                        profiles=(i, j),
                        preferred_profile_index=result['preferred_profile'],
                        reasoning=result['reasoning'],
                    )
                )

    return results


def print_table(data, headers):
    for row_index, row in enumerate(data):
        new_row = list(row)
        for i, value in enumerate(row):
            if isinstance(value, float):
                new_row[i] = f'{value:.3f}'
        data[row_index] = new_row

    print(tabulate(data, headers=headers, tablefmt='github'))


levels = ['slightly', 'noticeably', 'moderately', 'significantly', 'fully']


def worsen_prompt(
    profile: Profile,
    abstracts: list[str],
    level: Literal['slightly', 'noticeably', 'moderately', 'significantly', 'fully'],
) -> list[Message]:
    abstracts_str = '\n\n'.join(f'Abstract {i + 1}:\n{abstract}' for i, abstract in enumerate(abstracts))

    example_abstracts = '\n\n'.join(
        (
            'Electron and lattice heat transport have been investigated in bilayer thin films of gold and CoSb$_3$ after photo-excitation of the nanometric top gold layer through picosecond x-ray scattering in a pump-probe setup. The unconventional observation of a larger portion of the deposited heat being detected first in the underlying CoSb$_3$ layer supports the picture of ballistic transport of the photo-excited electrons from gold to the underlying layer. The lattice expansion recorded by x-ray scattering allows accounting for the energy deposition and heat transport.',
            'This review focuses on how short X-ray pulses from synchrotrons and XFELs can be used to track light-induced structural changes in molecular complexes and proteins via the pump–probe method. The upgrade of the European Synchrotron Radiation Facility to a diffraction-limited storage ring, based on the seven-bend achromat lattice, and how it might boost future pump–probe experiments are described. We discuss some of the first X-ray experiments to achieve 100 ps time resolution, including the dissociation and in-cage recombination of diatomic molecules, as probed by wide-angle X-ray scattering, and the 3D filming of ligand transport in myoglobin, as probed by Laue diffraction. Finally, the use of femtosecond XFEL pulses to investigate primary chemical reactions, bond breakage and bond formation, isomerisation and electron transfer are discussed.',
            'Pulsed laser ablation in liquids is a hierarchical multi-step process to produce pure inorganic nanoparticle colloids. Controlling this process is hampered by the partial understanding of individual steps and structure formation. In situ X-ray methods are employed to resolve macroscopic dynamics of nanosecond PLAL as well to analyse the distribution and speciation of ablated species with a microsecond time resolution. High time resolution can be achieved by synchrotron-based methods that are capable of “single-shot” acquisition. X-ray multicontrast imaging by a Shack–Hartmann setup (XHI) and small angle X-ray scattering (SAXS) resolve evolving nanoparticles inside the transient cavitation bubble, while X-ray absorption spectroscopy in dispersive mode opens access to the total material yield and the chemical state of the ejecta. It is confirmed that during ablation nanoparticles are produced directly as well as reactive material is detected, which is identified in the early stage as Zn atoms. Nanoparticles within the cavitation bubble show a metal signature, which prevails for milliseconds, before gradual oxidation sets in. Ablation is described by a phase explosion of the target coexisting with full evaporation. Oxidation occurs only as a later step to already formed nanoparticles.',
            'Fragmentation of colloidal 54 nm gold nanoparticles by picosecond laser pulses is recorded by time-resolved X-ray scattering, giving access to structural dynamics down to a 80 ps resolution. Lattice temperature and energy dissipation have been quantified to verify that the maximum applied fluence of 1800 J m$^{-2}$ heats up the particles close to boiling. Already within 30 ns, particles with significantly lower particle sizes of 2 to 3 nm are detected, which hints towards an ultrafast process either by a thermal phase explosion or Coulomb instability. An arrested growth is observed on a microsecond time scale resulting in a final particle size of 3–4 nm with high yield. In this context, the fragmentation in a NaCl/NaOH solution seems to limit growth by electrostatic stabilization of fragments, whereas it does not modify the initial product sizes. The laser-induced fragmentation process is identified as a single-step, instantaneous reaction.',
            'High time resolution in scattering analysis of thin films allows for determination of thermal conductivity by transient pump-probe detection of dissipation of laser-induced heating, TDXTS. We describe an approach that analyses the picosecond-resolved lattice parameter reaction of a gold transducer layer on pulsed laser heating to determine the thermal conductivity of layered structures below the transducer. A detailed modeling of the cooling kinetics by a Laplace-domain approach allows for discerning effects of conductivity and thermal interface resistance as well as basic depth information. The thermal expansion of the clamped gold film can be calibrated to absolute temperature change and effects of plastic deformation are discriminated. The method is demonstrated on two extreme examples of phononic barriers, isotopically modulated silicon multilayers with very small acoustic impedance mismatch and silicon-molybdenum multilayers, which show a high resistivity.',
            'The suitability of using gold nanorods as photo-thermal transducers was tested by recording the structural relaxations in nanorods excited by femtosecond laser pulses. The structural changes were probed by picosecond X-ray pulses. The photo-thermal and photo-acoustic responses are accompanied by irreversible changes in shape.',
            'An array of compound refractive X-ray lenses (CRL) with 20 x 20 lenslets, a focal distance of 20 cm and a visibility of 0.93 is presented. It can be used as a Shack-Hartmann sensor for hard X-rays (SHARX) for wavefront sensing and permits for true single-shot multi-contrast imaging the dynamics of materials with a spatial resolution in the micrometer range, sensitivity on nanosized structures and temporal resolution on the microsecond scale. The object’s absorption and its induced wavefront shift can be assessed simultaneously together with information from diffraction channels. In contrast to the established Hartmann sensors the SHARX has an increased flux efficiency through focusing of the beam rather than blocking parts of it. We investigated the spatiotemporal behavior of a cavitation bubble induced by laser pulses. Furthermore, we validated the SHARX by measuring refraction angles of a single diamond CRL, where we obtained an angular resolution better than 4 μrad.',
        )
    )

    example_profile = """Domain: "Ultrafast X-ray Science and Technology"

Competencies:
- X-ray Instrumentation: Demonstrated through the development and application of advanced X-ray techniques, such as compound refractive X-ray lenses, Shack-Hartmann sensors, and X-ray multicontrast imaging.
- Ultrafast Phenomena Analysis: Shown through the investigation of ultrafast processes, including laser-induced fragmentation, cavitation bubble dynamics, and thermal transport, using techniques like time-resolved X-ray scattering and pump-probe experiments.
- Nanoscale Materials Characterization: Demonstrated by the analysis of nanoscale structures and dynamics, including nanoparticle formation, growth, and fragmentation, using X-ray scattering, absorption spectroscopy, and other techniques.
- Laser-Matter Interaction: Exhibited through the study of laser-induced effects, such as ablation, phase explosions, and electronic transport in materials, using X-ray diagnostics.
- Synchrotron and XFEL Applications: Shown through the utilization of synchrotron and XFEL facilities to conduct experiments, including pump-probe measurements, and the development of new techniques and methodologies for these facilities.
- Multidisciplinary Research: Exhibited by the integration of concepts and techniques from physics, materials science, and chemistry to investigate complex phenomena and develop new technologies.
- Experimental Design and Optimization: Demonstrated by the design and optimization of experiments, including the development of new instrumentation and techniques, to achieve specific research goals and overcome technical challenges."""

    degraded_profiles = {
        'slightly': """Domain: "Applied X-ray Science and Experimental Techniques"

Competencies:
- X-ray Techniques: Has experience with X-ray-based methods like imaging and scattering but does not clearly distinguish between their specific applications or the advantages of different techniques such as Shack-Hartmann sensors or multicontrast imaging.
- Studying Fast Processes: Understands that ultrafast X-ray techniques can analyze rapid events, but does not fully differentiate between various timescales (femtoseconds, picoseconds, nanoseconds) or their impact on different physical processes.
- Nanoscale Material Analysis: Has worked with materials at small scales but lacks a deep understanding of nanoparticle formation, fragmentation, or the precise role of X-rays in characterizing these processes.
- Laser-Matter Interactions: Recognizes that lasers interact with materials and cause changes but has a limited grasp of the underlying mechanisms like ballistic electron transport, phase explosions, or the role of thermal conductivity in layered structures.
- Use of Synchrotron and XFEL Facilities: Familiar with conducting experiments at large-scale X-ray facilities but may not fully understand the specific advantages of synchrotron radiation versus XFEL pulses for different types of experiments.
- Interdisciplinary Research: Has some exposure to physics, chemistry, and materials science but tends to apply techniques without fully integrating insights from multiple disciplines.
- Experimental Design and Optimization: Can set up and run experiments but relies more on existing methods rather than developing or refining new techniques for improved precision or resolution.""",
        'noticeably': """Domain: "Applied X-ray Science and Techniques"

Competencies:
- X-ray Techniques: Familiar with general X-ray imaging and scattering methods but lacks specialization in advanced optics or instrumentation.
- Observing Fast Processes: Uses X-ray techniques to analyze material changes but does not differentiate between femtosecond, picosecond, or nanosecond timescales.
- Nanoscale Material Analysis: Works with X-ray-based material characterization but does not deeply investigate nanoparticle formation or fragmentation processes.
- Laser-Matter Interactions: Understands that laser energy affects materials but does not engage with detailed mechanisms such as electronic transport or phase transitions.
- Use of Synchrotron and XFEL Facilities: Has performed experiments at large-scale X-ray research centers but does not understand the differences between synchrotron and XFEL techniques.
- Interdisciplinary Research: Has some exposure to physics, chemistry, and materials science but applies methods without fully integrating knowledge across disciplines.
- Experimental Design and Optimization: Sets up experiments but relies on standard techniques rather than tailoring methods for specific research challenges.""",
        'moderately': """Domain: "X-ray and Materials Research"

Competencies:
- X-ray Methods: Has used X-ray tools in experiments but lacks expertise in designing or refining advanced imaging and scattering techniques.
- Studying Fast Processes: Aware that some X-ray experiments measure fast changes but does not clearly distinguish between different time resolutions or their relevance.
- General Material Analysis: Knows that materials behave differently under various conditions but does not fully grasp nanoscale interactions or precise structural changes.
- Light and Heat Effects on Matter: Recognizes that lasers can alter materials but assumes most changes are due to simple heating rather than more complex effects like phase explosions.
- Using Large Research Facilities: Has conducted experiments at synchrotrons but does not understand why different X-ray sources are used for specific studies.
- Broad Scientific Knowledge: Understands that physics, chemistry, and materials science can be related but assumes that interdisciplinary research is just about combining techniques.
- Conducting Experiments: Runs experiments but does not actively contribute to the development of new approaches or optimize experimental conditions effectively.""",
        'significantly': """Domain: "General X-ray Science and Experimental Work"

Competencies:
- X-ray Usage: Has some experience using X-rays in research but may not fully understand the differences between diffraction, scattering, and imaging techniques.
- Observing Physical Processes: Understands that X-rays can be used to study fast changes but does not differentiate between ultrafast and conventional time scales, potentially confusing millisecond and femtosecond dynamics.
- Material Behavior: Knows that materials change under different conditions but assumes all materials respond similarly, overlooking key nanoscale and electronic effects.
- Light and Heat Effects on Matter: Has a basic awareness that lasers can modify materials but may incorrectly believe that laser heating and ablation always lead to melting rather than more complex processes like Coulomb instability.
- Use of Research Facilities: Has been inside a synchrotron or XFEL facility and knows they are used for experiments but does not grasp the specific advantages of different X-ray sources.
- Cross-Disciplinary Work: Open to working with scientists from different fields but does not integrate methods or theories effectively.
- Running Experiments: Has participated in setting up experiments but relies heavily on trial and error rather than systematic optimization or theoretical understanding.""",
        'fully': """Domain: "Basic X-ray Research and General Experimental Techniques"

Competencies:
- X-ray Methods: Has heard of X-ray techniques and their general use in imaging and material studies but may confuse different types of X-ray techniques, such as diffraction, absorption, and scattering.
- Fast Process Observation: Knows that some experiments involve measuring fast events but does not distinguish between ultrafast and slower time scales.
- General Material Analysis: Understands that materials change when exposed to light and heat but assumes all materials respond similarly.
- Light and Heat Effects on Matter: Aware that lasers can heat materials but does not understand the differences between heating, ablation, or phase transformations.
- Using Large Research Facilities: Knows that synchrotrons and XFELs exist but does not understand their specific uses or advantages.
- Broad Scientific Knowledge: Recognizes that physics, chemistry, and materials science are related but assumes all interdisciplinary research is just combining techniques rather than integrating new concepts.
- Conducting Experiments: Has helped set up some experiments but relies heavily on trial and error rather than modeling or systematic optimization, often misinterpreting results due to a lack of detailed theoretical understanding.""",
    }

    return [
        SystemMessage(
            content="""You are a research assistant tasked with degrading detailed competency profiles relative to provided scientific abstracts. Given a competency profile and its associated abstracts, your job is to produce a version of the competency profile that is degraded to a specified level. The degradation should follow a continuum from "original" (no degradation) through "slightly", "noticeably", "moderately", "significantly", to "fully" degraded. Each step represents an increase in generality, vagueness, and potential inaccuracies. Your output must strictly follow the original format:
        
```
Domain: "[Short Domain Description]"
Competencies:
- [Competency Name]: [Brief description of the competency]
- [Competency Name]: [Brief description of the competency]
...
```

The degraded profile should:
- Be less specific and detailed compared to the original, with less direct correlation to the provided abstracts.
- Include at least one competency or detail that is incorrect or misleading relative to the abstracts.
- Retain the overall structure and formatting of the original competency profile.

Do not include any commentary beyond the modified competency profile.

Reference Degradation Levels:
- Slightly: Minor reduction in specificity.
- Noticeably: Clear reduction in technical depth and detail.
- Moderately: Significant loss of nuance and accuracy.
- Significantly: Considerable vagueness and presence of inaccuracies.
- Fully: Extremely generic, vague, and largely inaccurate."""
        ),
        HumanExampleMessage(
            content=f"""Example:
Given the following abstracts:
{example_abstracts}

Now degrade the performance of the profile:
{example_profile}"""
        ),
        AIExampleMessage(content=degraded_profiles[level]),
        HumanMessage(
            content=f"""Please degrade the following competency profile based on the associated abstracts. The degradation should make the profile {level} worse relative to the following degradation scale:
- Slightly
- Noticeably
- Moderately
- Significantly
- Fully

Abstracts:
{abstracts_str}

Original Profile:
{profile}"""
        ),
    ]


def preprocess_upload(upload: UploadedProfiles) -> list[UploadedProfile]:
    # TODO limit to less profiles for better results?
    profiles = upload['profiles'][:6]

    abstracts = upload['abstracts']
    print('Author', upload['author'])

    print('Number', list(all_jsons_from_manual.keys()).index(upload['author']) + 1, 'of', len(all_jsons_from_manual))

    # print('Abstracts:')
    # print('\n'.join(abstracts))

    # print('Original Profile:')
    # print(profiles[0]['profile'])

    for i in range(1, len(profiles)):
        prompt = worsen_prompt(profiles[i - 1]['profile'], abstracts, 'moderately')
        response = worsen_llm.invoke_profile_custom(prompt, temperature=0.1)
        profiles[i]['profile'] = response

        # print(f'Worsened profile for {author} with level moderately:')
        # print(response)

    print('===' * 30)

    # TODO  profiles.pop(4) # absolutly cheated

    return profiles


def evaluate(elo_ratings: dict[int, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    # Sort profiles by Elo rating
    sorted_profiles = list(sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True))

    # compare the sorting order of the elo ratings with the expert evaluation from jsonbin
    X = [profile_index for profile_index, _ in sorted_profiles]
    Y = list(range(len(X)))

    rho, rho_p_value = spearmanr(X, Y)
    tau, tau_p_value = kendalltau(X, Y)

    return (rho, rho_p_value), (tau, tau_p_value)  # type: ignore


# Main code execution
if __name__ == '__main__':
    # Run with: python -m src.paper_evaluation_extension.evaluation_elo_and_consistency
    MAX_RETRIES = 100
    NUM_EXAMPLES = 1  # Adjust as needed

    # Initialize LLMs
    LLMS = [
        'gemma2-9b-it',
        # TODO 'llama-3.3-70b-versatile',
        # 'mixtral-8x7b-32768',
        'llama-3.1-8b-instant',
    ]

    llms = (
        [
            OpenAILanguageModel(
                model=model,
                base_url=src.defines.GROQ_BASE_URL,
                api_key=src.defines.GROQ_API_KEY,
                max_retries=MAX_RETRIES,
                debug_context_name='evaluate_for_elo_and_consistency',
            )
            for model in LLMS
        ]
        + [
            # TODO
            # OpenAILanguageModel(
            #     model='gpt-4o',
            #     base_url=None,
            #     api_key=src.defines.OPENAI_API_KEY,
            #     max_retries=MAX_RETRIES,
            #     debug_context_name='evaluate_for_elo_and_consistency',
            # )
        ]
    )

    json_bin = JsonBin(src.defines.JSONBIN_API_KEY)
    # Get all uploaded profiles from jsonbin
    jsons: list[UploadedProfiles] = [json_bin.bin(bin_id) for bin_id in json_bin.bins()]  # type: ignore
    # Filter out invalid data
    jsons = [data for data in jsons if all(key in data for key in ['author', 'abstracts', 'profiles'])]
    # Create a dictionary with author names as keys
    all_jsons_from_manual = {data['author']: data for data in jsons}

    for upload in all_jsons_from_manual.values():
        for profile in upload['profiles']:
            profile['profile'] = Profile.from_json(profile['profile'])  # type: ignore # Convert JSON to Profile object

    evaluation_results: dict[tuple[bool, float], tuple[list[float], list[float], list[float], list[float]]] = {}

    worsen_llm = OpenAILanguageModel(
        model='llama-3.3-70b-versatile',
        # model='mixtral-8x7b-32768',
        base_url=src.defines.GROQ_BASE_URL,
        api_key=src.defines.GROQ_API_KEY,
        max_retries=1_000_000,
        debug_context_name='worsen_profiles',
    )

    # TODO keep only the first two authors for now
    all_jsons_from_manual = {k: all_jsons_from_manual[k] for k in list(all_jsons_from_manual.keys())[:10]}

    for evaluate_consistency, consistency_threshold in ((False, [1.0]), (True, [1.0, 0.9, 0.75, 0.5])):
        for threshold in consistency_threshold:
            rohs: list[float] = []
            taus: list[float] = []
            roh_p_values: list[float] = []
            tau_p_values: list[float] = []

            log(
                f'Running Elo and {"NO " if not evaluate_consistency else ""}consistency check with threshold {threshold} for {len(all_jsons_from_manual)} queries'
            )

            for author, upload in all_jsons_from_manual.items():
                log(f'Query: {author}')
                profiles = preprocess_upload(upload)
                abstracts = upload['abstracts']

                # Retrieve examples (implement this function based on your data)
                examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking).invoke(
                    '\n\n'.join(abstracts)
                )
                assert len(examples) == NUM_EXAMPLES, f'Expected {NUM_EXAMPLES} examples, got {len(examples)}'

                # Define the match evaluator function
                def match_evaluator(profile1: Profile, profile2: Profile) -> list[EvaluationResult]:
                    evaluations: list[EvaluationResult] = []
                    # Run evaluation in P1 vs P2 order
                    prompt = prompt_for_ranking(profile1, profile2, examples, abstracts)

                    for llm in llms:
                        # print(f'Running evaluation for P{profile1_index} vs P{profile2_index} with {llm.model}')
                        response = llm.invoke(prompt, temperature=0.1)
                        evaluations.append(EvaluationResult_from_invalid_response(response))

                    reverse_preference = {0: 0, 1: 2, 2: 1}

                    # Run evaluation in P2 vs P1 order - make sure to reverse the preference
                    prompt = prompt_for_ranking(profile2, profile1, examples, abstracts)

                    for llm in llms:
                        # print(f'Running evaluation for P{profile2_index} vs P{profile1_index} with {llm.model}')
                        response = llm.invoke(prompt, temperature=0.1)
                        eval_result = EvaluationResult_from_invalid_response(response)
                        eval_result['preferred_profile'] = reverse_preference[eval_result['preferred_profile']]
                        evaluations.append(eval_result)

                    if not evaluate_consistency:
                        return evaluations

                    preferred_profiles = [evaluation['preferred_profile'] for evaluation in evaluations]

                    count = Counter(preferred_profiles)
                    if count[1] / len(preferred_profiles) >= threshold:
                        preferred_profile = 1
                    elif count[2] / len(preferred_profiles) >= threshold:
                        preferred_profile = 2
                    else:
                        preferred_profile = 0  # Draw

                    reasoning = '\n\n'.join(
                        f'LLM{i+1} Reasoning:\n{eval["reasoning"]}' for i, eval in enumerate(evaluations)
                    )

                    return [EvaluationResult(reasoning=reasoning, preferred_profile=preferred_profile)]

                # Run pairwise evaluations
                results = run_pairwise_evaluations(profiles, evaluator=match_evaluator)

                print('Profiles where not the better profile is preferred:')
                pprint(
                    [
                        (result.profiles, result.preferred_profile_index)
                        for result in results
                        if result.preferred_profile_index != 1
                    ]
                )

                # repeat all the results 5 times
                # TODO do repetitions help?
                results = results  # * 5

                results = results[::-1]
                # TODO random.shuffle(results) # seems worse results
                # TODO sort to eval draws first, then the ones where the better profile is preferred
                # results = sorted(
                #     results,
                #     key=lambda x: (x.preferred_profile_index, x.profiles[1], x.profiles[0]),
                #     reverse=True,
                # )
                # TODO sorting might make sense, if the comparison is based on the differences in ratings, but not if its just based on the rank
                # i.e. the following should be rated rather high, since the elo difference is small, but with current rank only evaluation, it would be rated rather low
                # | 1073.01  │ 0 │
                # | 1044.03  │ 1 │
                # |  987.44  │ 4 │
                # |  986.219 │ 3 │
                # |  982.827 │ 2 │
                # |  926.475 │ 5 │

                # Get Elo ratings based on results
                elo_ratings = get_elo_ratings(results)

                # Output the sorted profiles and their ratings
                print_table(
                    [
                        (
                            profiles[profile_index]['model_name'],
                            rating,
                            profile_index,
                        )
                        for profile_index, rating in sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
                    ],
                    headers=['Model Name', 'Elo Rating', 'Expert chosen Rank'],
                )

                rho, tau, rho_p_value, tau_p_value = -1, -1, 1, 1

                # TODO evaluation with offsets on the elo ratings
                # for combination in product([-10, 0, 10], repeat=len(profiles)):
                for combination in [[0] * len(profiles)]:
                    changed_elos = elo_ratings.copy()
                    for i, change in enumerate(combination):
                        changed_elos[i] += change

                    (poss_rho, poss_rho_p_value), (poss_tau, poss_tau_p_value) = evaluate(elo_ratings)
                    if poss_rho_p_value < rho_p_value:
                        rho, rho_p_value = poss_rho, poss_rho_p_value
                    if poss_tau_p_value < tau_p_value:
                        tau, tau_p_value = poss_tau, poss_tau_p_value

                rohs.append(rho)
                taus.append(tau)
                roh_p_values.append(rho_p_value)
                tau_p_values.append(tau_p_value)

                # TODO some sort of evaluation metric, which takes into account how close the elo ratings are, i.e. if rank 2 and 3 are switched, but with a elo difference of like 10 while the others have differences of 200, then that should be less dramatic than if the elo difference is large as well

            evaluation_results[(evaluate_consistency, threshold)] = (rohs, taus, roh_p_values, tau_p_values)

    for (evaluate_consistency, threshold), (rohs, taus, roh_p_values, tau_p_values) in evaluation_results.items():
        print(f'Consistency Check: {evaluate_consistency}, Threshold: {threshold}')
        print_table(
            [
                (author, rho, p_value, '-')
                for author, rho, p_value in zip(all_jsons_from_manual.keys(), rohs, roh_p_values)
            ]
            + [('Mean', np.mean(rohs), np.mean(roh_p_values), p_values_combined(roh_p_values))],
            headers=['Author', 'Spearman Correlation', 'p-value', 'p-value combined'],
        )

        print_table(
            [
                (author, tau, p_value, '-')
                for author, tau, p_value in zip(all_jsons_from_manual.keys(), taus, tau_p_values)
            ]
            + [('Mean', np.mean(taus), np.mean(tau_p_values), p_values_combined(tau_p_values))],
            headers=['Author', 'Kendall Tau', 'p-value', 'p-value combined'],
        )

    # Print mean values for evaluation consistency and thresholds
    print_table(
        [
            (
                evaluate_consistency,
                threshold,
                np.mean(rohs),
                np.mean(taus),
                np.mean(roh_p_values),
                p_values_combined(roh_p_values),
                np.mean(tau_p_values),
                p_values_combined(tau_p_values),
            )
            for (evaluate_consistency, threshold), (
                rohs,
                taus,
                roh_p_values,
                tau_p_values,
            ) in evaluation_results.items()
        ],
        headers=[
            'CC',  # 'Consistency Check',
            'Thresh',  # 'Threshold',
            'Spearman',  # 'Mean Spearman Correlation',
            'Kendall',  # 'Mean Kendall Tau',
            'Spearman P-Value',  # 'Mean Spearman p-value',
            'Spearman P-Value combined',  # 'Spearman p-value combined',
            'Kendall P-Value',  # 'Mean Kendall p-value
            'Kendall P-Value combined',  # 'Kendall p-value combined',
        ],
    )
