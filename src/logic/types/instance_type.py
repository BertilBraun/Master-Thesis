from __future__ import annotations


from typing import Callable
from dataclasses import dataclass

from src.logic.types.base_types import Profile
from src.logic.types.database_types import RetrieverGetter
from src.logic.types.language_model_type import LanguageModel


@dataclass(frozen=True)
class Query:
    full_texts: list[str]
    abstracts: list[str]
    titles: list[str]
    author: str

    def __add__(self, other: Query) -> Query:
        assert self.author == other.author, 'Authors must be the same to combine queries'
        return Query(
            full_texts=self.full_texts + other.full_texts,
            abstracts=self.abstracts + other.abstracts,
            titles=self.titles + other.titles,
            author=self.author,
        )

    @staticmethod
    def from_json(data: dict) -> Query:
        return Query(
            full_texts=data['full_texts'],
            abstracts=data['abstracts'],
            titles=data['titles'],
            author=data['author'],
        )


@dataclass(frozen=True)
class Instance:
    # - Different Models (Types and Sizes)
    # - Abstract vs Automatic Summary vs Full Text
    # - Zero- vs One- vs Few-Shot
    # - TODO not yet - Good vs Bad Prompt
    # - TODO not supported by langchain - Good vs Bad Examples (best matches in VectorDB and worst matches in DB)

    model: str  # Identifier from OpenAI/Insomnium
    number_of_examples: int
    extract: Callable[[Query, RetrieverGetter, LanguageModel], Profile]

    @staticmethod
    def empty_instance() -> Instance:
        return Instance(
            model='',
            number_of_examples=0,
            extract=lambda q, r, l: Profile(domain='', competencies=[]),  # noqa
        )

    def __str__(self) -> str:
        return f'Instance({self.model} ({self.number_of_examples} examples) - {self.extract.__name__})'
