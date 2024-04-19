from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue

from src.log import LogLevel, log
from src.types import Profile, Example, LanguageModel
from src.database import DB


class OpenAILanguageModel(LanguageModel):
    def __init__(self, model: str):
        # Note: Cannot chain them because we need to pass the stop tokens to the model
        self.model = ChatOpenAI(model=model)
        self.output_parser = StrOutputParser()

    def batch(self, prompts: list[ChatPromptValue], /, stop: list[str] | None = None) -> list[str]:
        log(f'Running model: {self.model.name}', level=LogLevel.DEBUG)
        log(f'Prompts: {prompts}', level=LogLevel.DEBUG)
        response = self.output_parser.batch(self.model.batch(prompts, stop=stop))  # type: ignore
        log(f'Responses: {response}', level=LogLevel.DEBUG)
        return response

    def invoke(self, prompt: ChatPromptValue, /, stop: list[str] | None = None) -> str:
        return self.batch([prompt], stop=stop)[0]

    def invoke_profile(self, prompt: ChatPromptValue) -> Profile:
        stop = None  # TODO stop tokens ['\n\n\n']
        profile = Profile.parse(self.invoke(prompt, stop=stop))

        # TODO maybe not here
        DB.add(Example(abstract=str(prompt), profile=profile), author=self.model.name or 'unknown')

        return profile
