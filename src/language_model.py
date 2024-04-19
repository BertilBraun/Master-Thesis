from openai import OpenAI

from src.log import LogLevel, log
from src.types import Profile, Example, LanguageModel, Message
from src.database import add_element_to_database


class OpenAILanguageModel(LanguageModel):
    def __init__(self, model: str):
        self.model = model
        self.openai = OpenAI()

    def batch(self, prompts: list[list[Message]], /, stop: list[str] | None = None) -> list[str]:
        log(f'Running model: {self.model}', level=LogLevel.DEBUG)
        log(f'Prompts: {prompts}', level=LogLevel.DEBUG)

        results: list[str] = []

        for prompt in prompts:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[message.to_dict() for message in prompt],
                stop=stop,
                stream=False,
                temperature=0.0,  # TODO play with this?
            )

            result = response.choices[0].message.content or 'Error: No response from model'
            results.append(result)

        log(f'Responses: {results}', level=LogLevel.DEBUG)
        return results

    def invoke(self, prompt: list[Message], /, stop: list[str] | None = None) -> str:
        return self.batch([prompt], stop=stop)[0]

    def invoke_profile(self, prompt: list[Message]) -> Profile:
        stop = None  # TODO stop tokens ['\n\n\n']
        profile = Profile.parse(self.invoke(prompt, stop=stop))

        # TODO maybe not here
        add_element_to_database(Example(abstract=str(prompt), profile=profile))

        return profile
