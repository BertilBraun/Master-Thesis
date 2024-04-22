import src.openai_defines  # noqa # sets the OpenAI API key and base URL to the environment variables

from openai import OpenAI

from src.log import LogLevel, log
from src.types import Profile, Example, LanguageModel, Message
from src.database import add_element_to_database


DEBUG = True


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
                stream=DEBUG,
                temperature=0.0,  # TODO play with this?
            )

            if DEBUG:
                result = ''
                for chunk in response:  # type: ignore
                    delta = chunk.choices[0].delta.content or ''  # type: ignore
                    print(delta, end='', flush=True)
                    result += delta
            else:
                result = response.choices[0].message.content or 'Error: No response from model'  # type: ignore

            log(f'Response: {result}', level=LogLevel.DEBUG)
            results.append(result)

        log('------------------- End of batch -------------------', level=LogLevel.DEBUG)
        return results

    def invoke(self, prompt: list[Message], /, stop: list[str] = []) -> str:
        return self.batch([prompt], stop=stop)[0]

    def invoke_profile(self, prompt: list[Message]) -> Profile:
        stop = []  # TODO stop tokens ['\n\n\n']
        profile = Profile.parse(self.invoke(prompt, stop=stop))

        # TODO maybe not here
        add_element_to_database(Example(abstract=str(prompt), profile=profile), is_reference=False)

        return profile


def _format_prompt(prompt: list[Message], /) -> str:
    ret = ''
    for message in prompt:
        d = message.to_dict()
        ret += f'{d["role"]}: {d["content"]}\n'  # type: ignore

    return ret
