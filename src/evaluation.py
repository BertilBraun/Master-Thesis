from src.database import get_evaluation_messages, get_retriever_getter
from src.types import Evaluation, EvaluationResult, ExtractedProfile, Query, SystemMessage, HumanMessage
from src.language_model import OpenAILanguageModel


def evaluate_with(
    model: str,
    query: Query,
    extractions: list[ExtractedProfile],
) -> list[EvaluationResult]:
    llm = OpenAILanguageModel(model)

    retriever = get_retriever_getter(max_number_to_retrieve=1)(Evaluation)

    abstracts = '\n\n'.join(query.abstracts)

    prompts = [
        [
            SystemMessage(
                content="""You are a skilled evaluator tasked with assessing the relevance of competency profiles relative to the provided scientific abstracts. Evaluate how well each profile reflects the competencies, themes, and expertise areas mentioned in the abstracts. Structure your response as follows:
```
Evaluation and Reasoning: [Your Evaluation and Reasoning]
Score: [Your Score]
```

The evaluation should:
- Discuss the alignment of the profile's competencies with the themes and expertise areas of the abstracts.
- Highlight specific competencies that are well-represented or lacking in relation to the abstract content.
- Comment on the overall coherence between the profile's domain and the abstracts' focus areas.

The score should:
- Be between 0 and 100, where 100 represents a perfect alignment and 0 indicates no relevance at all.

Your analysis should be detailed, citing specific elements from both the profile and the abstracts to support your evaluation."""
            ),
            # TODO get one high scoring and one low scoring profile as an example?
            *get_evaluation_messages(abstracts, retriever),
            HumanMessage(
                content=f'Please assess the following competency profile in terms of its relevance to these scientific abstracts and provide a relevance score.\n\nAbstracts: {abstracts} \n\nProfile Details:\n{extraction.profile}\n\nYour evaluation should include specific examples and reasoning, followed by a score between 0 to 100.'
            ),
        ]
        for extraction in extractions
    ]

    responses = llm.batch(prompts)

    results = [
        EvaluationResult(
            extraction=extraction,
            reasoning=Evaluation.parse_reasoning(response),
            score=Evaluation.parse_evaluation_score(response),
        )
        for extraction, response in zip(extractions, responses)
    ]

    return results
