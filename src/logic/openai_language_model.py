from time import sleep
import src.defines  # noqa # sets the OpenAI API key and base URL to the environment variables

import tiktoken

from typing import Literal

from openai import OpenAI, RateLimitError

from src.util.log import LogLevel, log
from src.logic.types import LanguageModel, Message


class OpenAILanguageModel(LanguageModel):
    def __init__(
        self,
        model: str,
        debug_context_name: str = '',
        base_url: str | None = src.defines.BASE_URL_LLM,
        api_key: str | None = src.defines.OPENAI_API_KEY,
        max_retries: int = src.defines.MAX_RETRIES,
    ):
        self.model = model
        self.openai = OpenAI(base_url=base_url, api_key=api_key)
        self.debug_context_name = debug_context_name
        self.max_retries = max_retries

    def _get_llm_config(self) -> dict:
        return {'base_url': self.openai.base_url.host}

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


def trim_text_to_token_length(text: str, desired_token_length: int, model_name: str = 'gpt-3.5-turbo') -> str:
    encoding = tiktoken.encoding_for_model(model_name)
    encoded = encoding.encode(text)
    if len(encoded) <= desired_token_length:
        return text

    # Trim the text to the desired token length
    return encoding.decode(encoded[:desired_token_length], errors='ignore')
