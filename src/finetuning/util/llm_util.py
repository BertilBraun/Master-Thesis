import sys
from typing import Any

from src.util.log import LogLevel, log
from src.defines import LOCAL_AI_CODER, LOCAL_AI_ML_PC, OPENAI_API_KEY
from src.logic.language_model import OpenAILanguageModel


def parse_llm_from_sysargs() -> OpenAILanguageModel:
    # The script should be called as follows:
    # python script.py <model_id> <base_url> <api_key>
    # python script.py <endpoint>

    def parse_sys_arg(arg_name: str, arg_value: str, valid_values: dict[str, Any]) -> Any:
        try:
            return valid_values[arg_value]
        except KeyError:
            assert False, f'Invalid {arg_name} value: {arg_value}'

    # if sysargs are not provided, exit with error message
    if len(sys.argv) == 2:
        model_id, base_url, api_key = parse_sys_arg(
            'endpoint',
            sys.argv[1],
            {
                'openai': ('gpt-4o-mini', None, OPENAI_API_KEY),
                'mlpc': ('dev-llama-3-large', LOCAL_AI_ML_PC, None),
            },
        )
    elif len(sys.argv) == 4:
        # Model to use from sysargs
        model_id = parse_sys_arg('Model ID', sys.argv[1], {'gpt-4o': 'gpt-4o-mini', 'llama': 'dev-llama-3-large'})

        # Base url from sysargs either None or LOCAL_AI_ML_PC
        base_url = parse_sys_arg(
            'Base URL', sys.argv[2], {'openai': None, 'mlpc': LOCAL_AI_ML_PC, 'coder': LOCAL_AI_CODER}
        )

        # API key to use from sysargs
        api_key = parse_sys_arg('API Key', sys.argv[3], {'CAS': OPENAI_API_KEY, 'none': None})

    else:
        log(
            'Invalid number of arguments. Please provide either 1 argument for the endpoint or 3 arguments for the model_id, base_url, and api_key.\n\nCorrect usage:\npython script.py <model_id> <base_url> <api_key>\npython script.py <endpoint>',
            level=LogLevel.ERROR,
        )
        exit(1)

    return OpenAILanguageModel(
        model_id,
        debug_context_name='evaluate_samples_via_api',
        base_url=base_url,
        api_key=api_key,
        max_retries=2,
    )
