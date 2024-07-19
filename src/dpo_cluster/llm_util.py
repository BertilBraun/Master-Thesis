import sys
from typing import Any

from src.log import LogLevel, log
from src.defines import LOCAL_AI_CODER, LOCAL_AI_ML_PC
from src.dpo_cluster.defines import CAS_OPENAI_API_KEY
from src.language_model import OpenAILanguageModel


def parse_llm_from_sysargs() -> OpenAILanguageModel:
    # The script should be called as follows:
    # python script.py <model_id> <base_url> <api_key>

    # if sysargs are not provided, exit with error message
    if len(sys.argv) < 4:
        log(
            'Please provide the Model ID and the base URL and which API KEY to use as command line arguments.',
            level=LogLevel.ERROR,
        )
        exit(1)

    def parse_sys_arg(arg_name: str, arg_value: str, valid_values: dict[str, Any]) -> Any:
        INVALID_ARG_VALUE = f'Invalid {arg_name} value'
        value = valid_values.get(arg_value, INVALID_ARG_VALUE)
        assert value != INVALID_ARG_VALUE, f'{INVALID_ARG_VALUE}: {arg_value}'
        return value

    # Model to use from sysargs
    model_id = parse_sys_arg('Model ID', sys.argv[1], {'gpt-4o': 'gpt-4o-mini', 'llama': 'dev-llama-3-large'})

    # Base url from sysargs either None or LOCAL_AI_ML_PC
    base_url = parse_sys_arg('Base URL', sys.argv[2], {'openai': None, 'mlpc': LOCAL_AI_ML_PC, 'coder': LOCAL_AI_CODER})

    # API key to use from sysargs
    api_key = parse_sys_arg('API Key', sys.argv[3], {'CAS': CAS_OPENAI_API_KEY, 'none': None})

    return OpenAILanguageModel(
        model_id,
        debug_context_name='evaluate_samples_via_api',
        base_url=base_url,
        api_key=api_key,
        max_retries=2,
    )
