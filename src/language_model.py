import src.openai_defines  # noqa # sets the OpenAI API key and base URL to the environment variables

import tiktoken
from typing import Literal

from openai import OpenAI

from src.log import LogLevel, log, time_str
from src.types import AIMessage, Profile, LanguageModel, Message
from src.display import generate_html_file_for_chat


class OpenAILanguageModel(LanguageModel):
    def __init__(self, model: str, debug_context_name: str = ''):
        self.model = model
        self.openai = OpenAI(base_url=src.openai_defines.BASE_URL_LLM)
        self.debug_context_name = debug_context_name

    def batch(
        self,
        prompts: list[list[Message]],
        /,
        stop: list[str] = [],
        response_format: Literal['text'] | Literal['json_object'] = 'text',
        temperature: float = 0.5,
    ) -> list[str]:
        log('------------------ Start of batch ------------------', level=LogLevel.DEBUG)
        results = [
            self.invoke(prompt, stop=stop, response_format=response_format, temperature=temperature)
            for prompt in prompts
        ]
        log('------------------- End of batch -------------------', level=LogLevel.DEBUG)
        return results

    def invoke(
        self,
        prompt: list[Message],
        /,
        stop: list[str] = [],
        response_format: Literal['text'] | Literal['json_object'] = 'text',
        temperature: float = 0.5,
    ) -> str:
        assert len(stop) <= 4, 'The maximum number of stop tokens is 4'
        assert len(prompt) > 0, 'The prompt must contain at least one message'

        log(f'Running model: {self.model}', level=LogLevel.DEBUG)
        log(f'Prompt:\n{[m.to_dict() for m in prompt]}', level=LogLevel.DEBUG)

        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[message.to_dict() for message in prompt],
            stop=stop,
            stream=src.openai_defines.DEBUG,
            temperature=temperature,  # TODO play with this?
            response_format={'type': response_format},
        )

        if src.openai_defines.DEBUG:
            result = ''
            for chunk in response:  # type: ignore
                delta = chunk.choices[0].delta.content or ''  # type: ignore
                print(delta, end='', flush=True)
                result += delta
            print('\n\n')
        else:
            result = response.choices[0].message.content or 'Error: No response from model'

        result = result.replace('<dummy32000>', '')
        if response_format == 'json_object':
            # result = result from first { to last } inclusive
            result = result[result.find('{') : result.rfind('}') + 1]

        generate_html_file_for_chat(
            [*prompt, AIMessage(content=result)],
            f'{self.model}_{self.debug_context_name}_{time_str()}',
        )

        log(f'Response: {result}', level=LogLevel.DEBUG)

        if len(result) < 30:
            log('Response is most likely empty. Check the chat history for more information.', level=LogLevel.WARNING)
        return result

    def invoke_profile_custom(
        self,
        prompt: list[Message],
        temperature: float = 0.5,
    ) -> Profile:
        return Profile.parse(self.invoke(prompt, temperature=temperature))

    def invoke_profile_json(
        self,
        prompt: list[Message],
        temperature: float = 0.5,
    ) -> Profile:
        return Profile.parse_json(
            self.invoke(prompt, response_format='json_object', stop=['\n\n'], temperature=temperature)
        )


def trim_text_to_token_length(text: str, desired_token_length: int, model_name: str = 'gpt-3.5-turbo') -> str:
    encoding = tiktoken.encoding_for_model(model_name)
    encoded = encoding.encode(text)
    if len(encoded) <= desired_token_length:
        return text

    # Trim the text to the desired token length
    return encoding.decode(encoded[:desired_token_length], errors='ignore')
