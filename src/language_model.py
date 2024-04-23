import src.openai_defines  # noqa # sets the OpenAI API key and base URL to the environment variables

from typing import Literal

from openai import OpenAI

from src.log import LogLevel, log, time_str
from src.types import AIMessage, Profile, LanguageModel, Message
from src.display import generate_html_file_for_chat


class OpenAILanguageModel(LanguageModel):
    def __init__(self, model: str):
        self.model = model
        self.openai = OpenAI(base_url=src.openai_defines.BASE_URL_LLM)

    def batch(
        self,
        prompts: list[list[Message]],
        /,
        stop: list[str] = [],
        response_format: Literal['text'] | Literal['json_object'] = 'text',
    ) -> list[str]:
        log('------------------ Start of batch ------------------', level=LogLevel.DEBUG)
        results = [self.invoke(prompt, stop=stop, response_format=response_format) for prompt in prompts]
        log('------------------- End of batch -------------------', level=LogLevel.DEBUG)
        return results

    def invoke(
        self,
        prompt: list[Message],
        /,
        stop: list[str] = [],
        response_format: Literal['text'] | Literal['json_object'] = 'text',
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
            temperature=0.2,  # TODO play with this?
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
        generate_html_file_for_chat([*prompt, AIMessage(content=result)], f'{self.model}_{time_str()}')

        log(f'Response: {result}', level=LogLevel.DEBUG)
        return result

    def invoke_profile_custom(self, prompt: list[Message]) -> Profile:
        return Profile.parse(self.invoke(prompt))

    def invoke_profile_json(self, prompt: list[Message]) -> Profile:
        return Profile.parse_json(self.invoke(prompt, response_format='json_object'))
