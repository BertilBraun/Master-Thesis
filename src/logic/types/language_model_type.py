from __future__ import annotations


from typing import Literal, Protocol, overload


from src.logic.types.base_types import Profile
from src.logic.types.message_types import Message


class LanguageModel(Protocol):
    debug_context_name: str

    def __init__(self, model: str, debug_context_name: str = ''):
        ...

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

    def invoke_profile_custom(self, prompt: list[Message]) -> Profile:
        ...

    def invoke_profile_json(self, prompt: list[Message]) -> Profile:
        ...
