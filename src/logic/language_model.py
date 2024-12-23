import json
import os
from time import sleep
from src.finetuning.logic.tokenizer import get_tokenizer
from src.finetuning.logic.model import generate, get_model, prompt_messages_to_str
import src.defines  # noqa # sets the OpenAI API key and base URL to the environment variables

import tiktoken

from typing import Literal, overload
from partialjson.json_parser import JSONParser

from openai import OpenAI, RateLimitError

from src.util.log import LogLevel, log, time_str
from src.logic.types import AIMessage, LanguageModel, Message
from src.logic.display import generate_html_file_for_chat
from src.util import generate_hashcode


class OpenAILanguageModel(LanguageModel):
    def __init__(
        self,
        model: str,
        debug_context_name: str = '',
        base_url: str | None = src.defines.BASE_URL_LLM,
        api_key: str | None = None,
        max_retries: int = src.defines.MAX_RETRIES,
    ):
        self.model = model
        self.openai = OpenAI(base_url=base_url, api_key=api_key)
        self.debug_context_name = debug_context_name
        self.max_retries = max_retries

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

        log(f'Running model: {self.model}', level=LogLevel.DEBUG)
        log(f'Prompt:\n{[m.to_dict() for m in prompt]}', level=LogLevel.DEBUG)

        key = json.dumps(
            {
                'model': self.model,
                'base_url': self.openai.base_url.host,
                'messages': generate_hashcode([message.to_dict() for message in prompt]),
                'stop': stop,
                'temperature': temperature,
                'response_format': response_format,
            }
        )
        if os.path.exists('llm.json.cache'):
            with open('llm.json.cache', 'r') as f:
                cache = json.load(f)
                if key in cache:
                    return cache[key]
        else:
            cache = {}

        success, result = self._invoke_with_retry(
            prompt,
            stop,
            response_format,
            temperature,
            self.max_retries,
        )

        if success:
            with open('llm.json.cache', 'w') as f:
                cache[key] = result
                json.dump(cache, f)

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

    def _invoke(
        self,
        prompt: list[Message],
        stop: list[str],
        response_format: Literal['text', 'json_object'],
        temperature: float,
    ) -> tuple[bool, str]:
        # Returns [success, result string or error message]
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[message.to_dict() for message in prompt],
                stop=stop,
                stream=src.defines.DEBUG,
                temperature=temperature,  # TODO play with this?
                response_format={'type': response_format},
                max_tokens=1024,
                # 100 seconds - should be enough for the model to respond (unless the backend is down)
                timeout=100,
            )
        except RateLimitError as e:
            log(
                f'Error: Rate limit exceeded for model {self.model} with debug context {self.debug_context_name}: {e}',
                level=LogLevel.WARNING,
            )
            sleep(60)
            return False, 'Error: Rate limit exceeded'
        except Exception as e:
            log(
                f'Error Generating Response: {e} for model {self.model} with debug context {self.debug_context_name}',
                level=LogLevel.WARNING,
            )
            return False, f'Error: Exception occurred while generating response {e}'

        if src.defines.DEBUG:
            result = ''
            for chunk in response:  # type: ignore
                delta = chunk.choices[0].delta.content or ''  # type: ignore
                print(delta, end='', flush=True)
                result += delta
            print('\n\n')
        else:
            if not response.choices or not response.choices[0].message.content:
                log(
                    f'Error: No response from model for model {self.model} with debug context {self.debug_context_name}',
                    level=LogLevel.WARNING,
                )
                return False, 'Error: No response from model'
            result = response.choices[0].message.content

        result = result.replace('<dummy32000>', '')
        result = result.replace('</s>', '')

        if len(result) < 30:
            log(
                'Response is most likely empty. Check the chat history for more information.',
                level=LogLevel.WARNING,
            )
            return False, result

        return True, result


class TransformersLanguageModel(LanguageModel):
    def __init__(
        self,
        model: str,
        debug_context_name: str = '',
        max_retries: int = src.defines.MAX_RETRIES,
    ):
        self.tokenizer = get_tokenizer(model)
        self.model = get_model(model)
        self.debug_context_name = debug_context_name
        self.max_retries = max_retries

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
        log(f'Running model: {self.model.name_or_path}', level=LogLevel.DEBUG)
        log(f'Prompt:\n{prompt}', level=LogLevel.DEBUG)

        prompt_str = prompt_messages_to_str(self.tokenizer, prompt)

        result = generate(
            self.tokenizer,
            self.model,
            prompt_str,
            num_return_sequences=1,
            do_sample=True,
            max_new_tokens=800,
            temperature=temperature,
            skip_special_tokens=True,
        )[0]

        return result if response_format == 'text' else json.loads(result)


def trim_text_to_token_length(text: str, desired_token_length: int, model_name: str = 'gpt-3.5-turbo') -> str:
    encoding = tiktoken.encoding_for_model(model_name)
    encoded = encoding.encode(text)
    if len(encoded) <= desired_token_length:
        return text

    # Trim the text to the desired token length
    return encoding.decode(encoded[:desired_token_length], errors='ignore')
