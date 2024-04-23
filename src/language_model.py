import src.openai_defines  # noqa # sets the OpenAI API key and base URL to the environment variables

from openai import OpenAI

from src.log import LogLevel, log, time_str
from src.types import AIMessage, Profile, LanguageModel, Message
from src.display import generate_html_file_for_chat


class OpenAILanguageModel(LanguageModel):
    def __init__(self, model: str):
        self.model = model
        self.openai = OpenAI(base_url=src.openai_defines.BASE_URL_LLM)

    def batch(self, prompts: list[list[Message]], /, stop: list[str] = []) -> list[str]:
        log(f'Running model: {self.model}', level=LogLevel.DEBUG)
        log('------------------ Start of batch ------------------', level=LogLevel.DEBUG)

        results: list[str] = []

        for prompt in prompts:
            log(f'Prompt: {_format_prompt(prompt)}', level=LogLevel.DEBUG)
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[message.to_dict() for message in prompt],
                stop=stop,
                stream=src.openai_defines.DEBUG,
                temperature=0.2,  # TODO play with this?
            )

            if src.openai_defines.DEBUG:
                result = ''
                for chunk in response:  # type: ignore
                    delta = chunk.choices[0].delta.content or ''  # type: ignore
                    print(delta, end='', flush=True)
                    result += delta
                print('\n\n')
            else:
                result = response.choices[0].message.content or 'Error: No response from model'  # type: ignore

            result = result.replace('<dummy32000>', '')
            generate_html_file_for_chat([*prompt, AIMessage(content=result)], f'{self.model}_{time_str()}')

            log(f'Response: {result}', level=LogLevel.DEBUG)
            results.append(result)

        log('------------------- End of batch -------------------', level=LogLevel.DEBUG)
        return results

    def invoke(self, prompt: list[Message], /, stop: list[str] = []) -> str:
        return self.batch([prompt], stop=stop)[0]

    def invoke_profile(self, prompt: list[Message]) -> Profile:
        stop = []  # TODO stop tokens ['\n\n\n']
        return Profile.parse(self.invoke(prompt, stop=stop))


def _format_prompt(prompt: list[Message], /) -> str:
    ret = ''
    for message in prompt:
        d = message.to_dict()
        ret += f'{d["role"]}: {d["content"]}\n'  # type: ignore

    return ret
