from src.language_model import trim_text_to_token_length
from src.types import (
    Combination,
    Example,
    Profile,
    Query,
    LanguageModel,
    RetrieverGetter,
    SystemMessage,
    HumanMessage,
)
from src.database import (
    get_combination_messages,
    get_example_messages,
)


def extract_from_abstracts_custom(query: Query, retriever: RetrieverGetter, llm: LanguageModel) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the abstracts
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    abstracts = '\n\n'.join(query.abstracts)

    prompt = [
        SystemMessage(
            content="""You are a helpful research assistant tasked with analyzing scientific abstracts to extract professional competencies. For each abstract, identify the primary domain of expertise and list specific competencies demonstrated by the author. Format your findings as follows:
```
Domain: [Short Domain Description]
Competencies:
- [Competency 1]: [Brief description of how Competency 1 is demonstrated across the abstracts]
- [Competency 2]: [Brief description of how Competency 2 is demonstrated across the abstracts]
...
```
Extract 3 to 8 competencies for each abstract, providing a clear and concise description for each. The domain description should be a brief label, summarizing the overall area of expertise. Your analysis should be neutral, accurate, and solely based on the content of the abstracts provided."""
        ),
        *get_example_messages(abstracts, retriever(Example)),
        HumanMessage(
            content=f'Please analyze these scientific abstracts and extract a single professional profile that reflects the competencies and domain of expertise demonstrated throughout. Consider the entire set of abstracts as one cohesive source for a comprehensive competency overview.\n\n{abstracts}'
        ),
    ]

    return llm.invoke_profile_custom(prompt)


def extract_from_summaries_custom(query: Query, retriever: RetrieverGetter, llm: LanguageModel) -> Profile:
    # We are putting all Papers in one Prompt but only looking at the summaries
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # First Stage: Generation of Summaries for each Paper

    # Get the summary from the full text
    prompts = [
        [
            SystemMessage(
                content="""You are a helpful research assistant.
For each scientific paper, generate a summary following this standardized structure: 
    1. Start with the title of the paper. 
    2. Provide a brief background or introduction to the study's context and primary question or hypothesis.
    3. State the objectives of the study.
    4. Summarize the methods, including experimental design, data collection, and analysis techniques.
    5. Highlight the main results, including key data and statistical outcomes.
    6. Summarize the discussion or conclusion, focusing on the interpretation of results, conclusions drawn, and their implications.
    7. Identify the key contributions of the paper to the field.
    8. Note any limitations of the study.
    9. End with any future research directions or suggestions provided by the authors. 

Please exclude redundant information such as authors, publication date, and location. Focus solely on the content of the paper."""
            ),
            # TODO do examples really help? Prompt is way too long already *get_summary_messages(full_text, retriever(Summary)),
            HumanMessage(
                content=f'Summarize the following paper.\nPaper:\n\n{trim_text_to_token_length(full_text, 6000)}\n\nNow generate a comprehensive summary of this paper that includes the title, background, objectives, methods, results, discussion, contributions, limitations, and future research directions.'
            ),
        ]
        for full_text in query.full_texts
    ]

    summaries = '\n\n'.join(llm.batch(prompts))

    # Second Stage: Extraction of Competencies from Summaries

    prompt = [
        SystemMessage(
            content="""You are a helpful research assistant tasked with extracting professional competencies from a set of summarized scientific papers. Review all the provided summaries comprehensively to create a unified professional profile that captures the overarching domain of expertise and specific competencies demonstrated across the texts. Format your consolidated findings as follows:
```
Domain: [Short Domain Description]
Competencies:
- [Competency 1]: [Brief description of how Competency 1 is demonstrated across the summaries]
- [Competency 2]: [Brief description of how Competency 2 is demonstrated across the summaries]
...
```
Identify and list 3 to 8 competencies, providing concise descriptions for each. The domain should succinctly summarize the general area of research. Ensure your analysis is neutral and precise, based solely on the content of the summaries provided. Consider the entire set of summaries as one cohesive source for a comprehensive competency overview."""
        ),
        *get_example_messages(summaries, retriever(Example)),
        HumanMessage(
            content=f'Please analyze these scientific paper summaries and extract a single professional profile that reflects the competencies and domain of expertise demonstrated throughout. Here are the summaries:\n\n{summaries}'
        ),
    ]

    return llm.invoke_profile_custom(prompt)


def _extract_from_full_texts_custom(
    query: Query, retriever: RetrieverGetter, llm: LanguageModel
) -> tuple[list[Profile], Profile]:
    # We are summarizing one Paper per Prompt, afterwards combining the extracted competences
    # NOTE: Throws AssertionError if the model is not able to generate a valid Profile from the papers

    # First Stage: Extraction of Individual Competency Profiles

    prompts = [
        [
            SystemMessage(
                content="""You are a helpful research assistant tasked with identifying and cataloging professional competencies from a scientific paper. Extract all relevant competencies demonstrated within the text, and organize them into a structured competency profile as follows:
```
Domain: [Short Domain Description]
Competencies:
- [Competency 1]: [Detailed explanation of how Competency 1 is demonstrated in the text]
- [Competency 2]: [Detailed explanation of how Competency 2 is demonstrated in the text]
...
```
List all pertinent competencies, clearly detailing how each is evidenced in the document. The domain should succinctly summarize the general area of research. Ensure your analysis is neutral and precise, based solely on the content of the paper provided."""
            ),
            *get_example_messages(full_text, retriever(Example)),
            HumanMessage(
                content=f"""Please extract the professional competencies from this complete document text:
                
{trim_text_to_token_length(full_text, 6000)}


The domain should succinctly summarize the general area of research of the paper. Then list all pertinent competencies, clearly detailing how each is evidenced in the document.  This is the format:
```
Domain: [Short Domain Description]
Competencies:
- [Competency 1]: [Detailed explanation of how Competency 1 is demonstrated in the text]
- [Competency 2]: [Detailed explanation of how Competency 2 is demonstrated in the text]
...
```
Ensure your analysis is neutral and precise, based solely on the content of the paper provided."""
            ),
        ]
        for full_text in query.full_texts
    ]

    llm_profiles = llm.batch(prompts)

    # Assuming conversion of profiles to string format and joining them happens here.
    profiles = [Profile.parse(profile) for profile in llm_profiles]
    profiles_str = '\n\n'.join(str(profile) for profile in profiles)

    # Second Stage: Combining Individual Profiles into a Comprehensive Profile

    prompt = [
        SystemMessage(
            content="""You are now tasked with synthesizing individual competency profiles into a single comprehensive profile. This unified profile should integrate and encapsulate the essence of all the individual profiles provided, formatted as follows:
```
Domain: [Consolidated Domain Description]
Competencies:
- [Integrated Competency 1]: [Consolidated description based on individual profiles]
- [Integrated Competency 2]: [Consolidated description based on individual profiles]
...
```
Combine the competencies into 3 to 8 competencies to reflect overarching skills and expertise demonstrated across all texts. The domain should succinctly summarize the general area of research over all profiles and competencies involved. Ensure your analysis is neutral and precise, based solely on the content of the summaries provided. Consider the entire set of summaries as one cohesive source for a comprehensive competency overview."""
        ),
        *get_combination_messages(profiles_str, retriever(Combination)),
        HumanMessage(
            content=f'Please synthesize these individual profiles into one comprehensive profile of 3 to 8 competencies:\n\n{profiles_str}'
        ),
    ]

    return profiles, llm.invoke_profile_custom(prompt)


def extract_from_full_texts_custom(query: Query, retriever: RetrieverGetter, llm: LanguageModel) -> Profile:
    profiles, comprehensive_profile = _extract_from_full_texts_custom(query, retriever, llm)
    return comprehensive_profile