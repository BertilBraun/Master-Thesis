from __future__ import annotations


import json
import os
from typing import Literal, Protocol, overload

from partialjson import JSONParser


from src.logic.display import generate_html_file_for_chat
from src.logic.types.base_types import Profile
from src.logic.types.message_types import AIMessage, Message
from src.util.cache import generate_hashcode
from src.util.log import LogLevel, log, time_str


class LanguageModel(Protocol):
    model: str
    debug_context_name: str
    max_retries: int

    def __init__(self, model: str, debug_context_name: str = '', max_retries: int = 1) -> None:
        ...

    @overload
    def batch(
        self,
        prompts: list[list[Message]],
        /,
        stop: list[str] = [],
        response_format: Literal['text'] = 'text',
        temperature: float = 0.5,
    ) -> list[str]:
        ...

    @overload
    def batch(
        self,
        prompts: list[list[Message]],
        /,
        stop: list[str] = [],
        response_format: Literal['json_object'] = 'json_object',
        temperature: float = 0.5,
    ) -> list[dict]:
        ...

    def batch(
        self,
        prompts: list[list[Message]],
        /,
        stop: list[str] = [],
        response_format: Literal['text', 'json_object'] = 'text',
        temperature: float = 0.5,
    ) -> list[str] | list[dict]:
        log('------------------ Start of batch ------------------', level=LogLevel.DEBUG)
        results = [
            self.invoke(prompt, stop=stop, response_format=response_format, temperature=temperature)
            for prompt in prompts
        ]
        log('------------------- End of batch -------------------', level=LogLevel.DEBUG)
        return results  # type: ignore

    @overload
    def invoke(
        self,
        prompt: list[Message],
        /,
        stop: list[str] = [],
        response_format: Literal['text'] = 'text',
        temperature: float = 0.5,
    ) -> str:
        ...

    @overload
    def invoke(
        self,
        prompt: list[Message],
        /,
        stop: list[str] = [],
        response_format: Literal['json_object'] = 'json_object',
        temperature: float = 0.5,
    ) -> dict:
        ...

    def invoke(
        self,
        prompt: list[Message],
        /,
        stop: list[str] = [],
        response_format: Literal['text', 'json_object'] = 'text',
        temperature: float = 0.5,
    ) -> str | dict:
        assert len(stop) <= 4, 'The maximum number of stop tokens is 4'
        assert len(prompt) > 0, 'The prompt must contain at least one message'

        key_parameters = {
            'messages': generate_hashcode([message.to_dict() for message in prompt]),
            'stop': stop,
            'temperature': temperature,
            'response_format': response_format,
        }

        if response := self._try_get_cached_response(**key_parameters):
            return response

        success, result = self._invoke_with_retry(
            prompt,
            stop,
            response_format,
            temperature,
            self.max_retries,
        )

        if success:
            self._cache_response(result, **key_parameters)

        return result

    def _invoke_with_retry(
        self,
        prompt: list[Message],
        stop: list[str],
        response_format: Literal['text', 'json_object'],
        temperature: float,
        retries: int,
    ) -> tuple[bool, str | dict]:
        # Returns [success, result string or error message or json object]
        if retries <= 0:
            log(
                f'Error: Maximum retries reached for model {self.model} with debug context {self.debug_context_name}! Is the backend down?',
                level=LogLevel.WARNING,
            )
            return False, 'Error: Maximum retries reached'

        success, result = self._invoke(prompt, stop, response_format, temperature)

        try_str = '' if retries == self.max_retries else f' (try {self.max_retries-retries+1})'
        generate_html_file_for_chat(
            prompt + [AIMessage(content=result)],
            f'{time_str()}_{self.model}_{self.debug_context_name}{try_str}',
        )

        if not success:
            if 'timeout' in result.lower():
                log('Backend seems to be down!', level=LogLevel.ERROR)
                exit(1)

            return self._invoke_with_retry(
                prompt,
                stop,
                response_format,
                temperature,
                retries - 1,
            )

        if response_format == 'json_object':
            # result = result from first { to last } inclusive
            # result = result[result.find('{') : result.rfind('}') + 1]
            try:
                result = JSONParser().parse(result)
            except Exception as e:
                log(
                    f'Error: Failed to parse JSON response for model {self.model} with debug context {self.debug_context_name}: {e}',
                    level=LogLevel.WARNING,
                )
                return False, 'Error: Failed to parse JSON response'

        log(f'Response: {result}', level=LogLevel.DEBUG)

        return True, result

    def invoke_profile_custom(
        self,
        prompt: list[Message],
        temperature: float = 0.5,
    ) -> Profile:
        for _ in range(self.max_retries):
            try:
                return Profile.parse(self.invoke(prompt, stop=['\n\n\n\n'], temperature=temperature))
            except AssertionError as e:
                log(e, level=LogLevel.WARNING)
                if 'not found in' not in str(e):
                    raise e

                self._remove_cached_response(
                    messages=generate_hashcode([message.to_dict() for message in prompt]),
                    stop=['\n\n\n\n'],
                    temperature=temperature,
                    response_format='text',
                )

        raise AssertionError('Failed to parse profile')

    def invoke_profile_json(
        self,
        prompt: list[Message],
        temperature: float = 0.5,
    ) -> Profile:
        for _ in range(self.max_retries):
            try:
                return Profile.parse_json(
                    self.invoke(prompt, response_format='json_object', stop=['\n\n\n\n'], temperature=temperature)
                )
            except AssertionError as e:
                log(e, level=LogLevel.WARNING)
                if 'not found in' not in str(e):
                    raise e

                self._remove_cached_response(
                    messages=generate_hashcode([message.to_dict() for message in prompt]),
                    stop=['\n\n\n\n'],
                    temperature=temperature,
                    response_format='json_object',
                )

        raise AssertionError('Failed to parse profile')

    def _get_key(self, /, **key_parameters) -> str:  # TODO -> int:
        return (  # TODO hash(
            json.dumps(
                {
                    'model': self.model,
                    **self._get_llm_config(),
                    **key_parameters,
                }
            )
        )

    def _try_get_cached_response(self, /, **key_parameters) -> str | dict | None:
        if os.path.exists('llm.json.cache'):
            with open('llm.json.cache', 'r') as f:
                cache = json.load(f)
                key = self._get_key(**key_parameters)
                if key in cache:
                    return cache[key]

        return None

        key = self._get_key(**key_parameters)
        if os.path.exists(f'llm_cache/{key}.json'):
            with open(f'llm_cache/{key}.json', 'r') as f:
                return json.load(f)

    def _cache_response(self, response: str | dict, /, **key_parameters) -> None:
        if os.path.exists('llm.json.cache'):
            with open('llm.json.cache', 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

        key = self._get_key(**key_parameters)
        cache[key] = response
        with open('llm.json.cache', 'w') as f:
            json.dump(cache, f)
        return

        key = self._get_key(**key_parameters)
        with open(f'llm_cache/{key}.json', 'w') as f:
            json.dump(response, f)

    def _remove_cached_response(self, /, **key_parameters) -> None:
        if os.path.exists('llm.json.cache'):
            with open('llm.json.cache', 'r') as f:
                cache = json.load(f)
            key = self._get_key(**key_parameters)
            if key in cache:
                del cache[key]
                with open('llm.json.cache', 'w') as f:
                    json.dump(cache, f)

        return
        key = self._get_key(**key_parameters)
        if os.path.exists(f'llm_cache/{key}.json'):
            os.remove(f'llm_cache/{key}.json')

    def _get_llm_config(self) -> dict:
        ...

    def _invoke(
        self,
        prompt: list[Message],
        stop: list[str],
        response_format: Literal['text', 'json_object'],
        temperature: float,
    ) -> tuple[bool, str]:
        ...
