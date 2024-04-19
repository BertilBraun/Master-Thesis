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

    def invoke(self, prompt: ChatPromptValue, /, stop: list[str] = []) -> str:
        log(f'Running model: {self.model.name}', level=LogLevel.DEBUG)
        log(f'Prompt: {prompt}', level=LogLevel.DEBUG)
        response = self.output_parser.invoke(self.model.invoke(prompt, stop=stop))
        log(f'Response: {response}', level=LogLevel.DEBUG)
        return response

    def invoke_profile(self, prompt: ChatPromptValue) -> Profile:
        # TODO stop tokens
        stop = ['\n\n\n']
        profile = Profile.parse(self.invoke(prompt, stop=stop))

        # TODO maybe not here
        DB.add(Example(abstract=str(prompt), profile=profile), author=self.model.name or 'unknown')

        return profile

    def batch(self, prompts: list[ChatPromptValue], /, stop: list[str] = []) -> list[str]:
        log(f'Running batched model: {self.model.name}', level=LogLevel.DEBUG)
        log(f'Prompts: {prompts}', level=LogLevel.DEBUG)
        response = self.output_parser.batch(self.model.batch(prompts, stop=stop))  # type: ignore
        log(f'Response: {response}', level=LogLevel.DEBUG)
        return response
