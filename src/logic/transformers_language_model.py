import json
from src.finetuning.logic.tokenizer import get_tokenizer
from src.finetuning.logic.model import generate, get_model, prompt_messages_to_str
import src.defines  # noqa # sets the OpenAI API key and base URL to the environment variables


from typing import Literal, overload


from src.util.log import LogLevel, log
from src.logic.types import LanguageModel, Message


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
