from src.extraction.extraction_custom import (
    extract_from_abstracts_custom,
    extract_from_full_texts_custom,
    extract_from_summaries_custom,
)
from src.extraction.extraction_json import (
    extract_from_abstracts_json,
    extract_from_full_texts_json,
    extract_from_summaries_json,
)

OTHER_REFERENCE_GENERATION_MODEL = 'alias-fast-instruct'
REFERENCE_GENERATION_MODEL = 'alias-fast-instruct'
EVALUATION_MODEL = 'alias-large-instruct'

DO_SHUFFLE_DURING_EVALUATION = True
RUN_EVALUATION = True

# Model defines based on https://gitlab.kit.edu/kit/aifb/BIS/infrastruktur/localai/localai-model-gallery
MODELS = [
    'dev-phi-3-mini',  # 3.8B parameters
    'dev-phi-3-medium',  # 14B parameters
    'alias-large-instruct',  # Mixtral 8x7B parameters
    # 'alias-fast-instruct',  # Hermes-2-Pro-Llama-3-Instruct 8B parameters
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
