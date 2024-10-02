from __future__ import annotations


from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessageParam


@dataclass(frozen=True)
class SystemMessage:
    content: str

    def to_dict(self) -> ChatCompletionMessageParam:
        return {'content': self.content, 'role': 'system'}


@dataclass(frozen=True)
class HumanMessage:
    content: str

    def to_dict(self) -> ChatCompletionMessageParam:
        return {'content': self.content, 'role': 'user'}


@dataclass(frozen=True)
class AIMessage:
    content: str

    def to_dict(self) -> ChatCompletionMessageParam:
        return {'content': self.content, 'role': 'assistant'}


@dataclass(frozen=True)
class HumanExampleMessage:
    content: str

    def to_dict(self) -> ChatCompletionMessageParam:
        return {'content': self.content, 'role': 'user'}  # Phi3mini does not support the name field
        return {'content': self.content, 'name': 'example_user', 'role': 'system'}


@dataclass(frozen=True)
class AIExampleMessage:
    content: str

    def to_dict(self) -> ChatCompletionMessageParam:
        return {'content': self.content, 'role': 'assistant'}  # Phi3mini does not support the name field
        return {'content': self.content, 'name': 'example_assistant', 'role': 'system'}


Message = SystemMessage | HumanMessage | AIMessage | HumanExampleMessage | AIExampleMessage
