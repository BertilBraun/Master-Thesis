import src.openai_defines  # noqa # sets the OpenAI API key and base URL to the environment variables

import os

from enum import Enum
from typing import Any, Protocol, Type, TypeVar

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.types import (
    Combination,
    Evaluation,
    Example,
    HumanExampleMessage,
    AIExampleMessage,
    Profile,
    Retriever,
    Message,
    RetrieverGetter,
    Summary,
)

DB_LOG_FOLDER = 'logs'
DB_LOG_FILE = DB_LOG_FOLDER + '/db.log'
os.makedirs(DB_LOG_FOLDER, exist_ok=True)

db = Chroma(
    persist_directory='data',
    collection_name='example',
    embedding_function=OpenAIEmbeddings(),
    collection_metadata={'reference': 'bool', 'type': 'string'},
)


class DBEntryType(Enum):
    EXAMPLE = 'example'
    SUMMARY = 'summary'
    EVALUATION = 'evaluation'
    COMBINATION = 'combination'


TYPE_TO_CLASS = {
    DBEntryType.EXAMPLE: Example,
    DBEntryType.SUMMARY: Summary,
    DBEntryType.EVALUATION: Evaluation,
    DBEntryType.COMBINATION: Combination,
}

CLASS_TO_TYPE = {v: k for k, v in TYPE_TO_CLASS.items()}


def _append_to_log(msg: str) -> None:
    with open(DB_LOG_FILE, 'a') as f:
        f.write(msg)


def add_element_to_database(element: Summary | Example | Evaluation | Combination, is_reference: bool = False) -> str:
    # Adds the element to the database and returns the ID of the added document

    metadata = {
        'reference': is_reference,
        'type': CLASS_TO_TYPE[type(element)].value,
    }

    ids = db.add_documents([Document(page_content=str(element), metadata=metadata)])

    _append_to_log(f'Added document to DB:\nPage content: {element}\nMetadata: {metadata}')

    return ids[0]


class Parsable(Protocol):
    @staticmethod
    def parse(text: str) -> Any:
        ...


T = TypeVar('T', bound=Parsable)


def get_database_as_retriever(max_number_to_retrieve: int, return_type: Type[T]) -> Retriever[T]:
    retriever = db.as_retriever(
        search_kwargs={
            'k': max_number_to_retrieve,
            'filter': {'type': CLASS_TO_TYPE[return_type].value},  # TODO also? "reference: True"
        }
    ).with_fallbacks([RunnableLambda(lambda query: [])])

    def parse_documents(docs: list[Document]) -> list[T]:
        return [return_type.parse(doc.page_content) for doc in docs]

    return retriever | parse_documents


def get_retriever_getter(max_number_to_retrieve: int) -> RetrieverGetter:
    def get_retriever(return_type: Type[T]) -> Retriever[T]:
        return get_database_as_retriever(max_number_to_retrieve, return_type)

    return get_retriever


def get_example_messages(content: str, retriever: Retriever[Example]) -> list[Message]:
    examples = retriever.invoke(content)

    return [
        message
        for example in examples
        for message in [
            HumanExampleMessage(content=example.abstract),
            AIExampleMessage(content=str(example.profile)),
        ]
    ]


def get_summary_messages(content: str, retriever: Retriever[Summary]) -> list[Message]:
    summaries = retriever.invoke(content)

    return [
        message
        for summary in summaries
        for message in [
            HumanExampleMessage(content=summary.full_text),
            AIExampleMessage(content=summary.summary),
        ]
    ]


def get_evaluation_messages(content: str, retriever: Retriever[Evaluation]) -> list[Message]:
    evaluations = retriever.invoke(content)

    return [
        message
        for evaluation in evaluations
        for message in [
            HumanExampleMessage(content=evaluation.text + '\n\n' + str(evaluation.profile)),
            AIExampleMessage(content=str(evaluation.score)),
        ]
    ]


def get_combination_messages(content: str, retriever: Retriever[Combination]) -> list[Message]:
    combinations = retriever.invoke(content)

    return [
        message
        for combination in combinations
        for message in [
            HumanExampleMessage(content='\n\n'.join(str(profile) for profile in combination.input_profiles)),
            AIExampleMessage(content=str(combination.output_profile)),
        ]
    ]


if __name__ == '__main__':
    from src.types import Profile, Example, Competency

    id1 = add_element_to_database(
        Example(
            abstract='harrison worked at kensho',
            profile=Profile(
                domain='Finance',
                competencies=[
                    Competency(name='Python', description='Proficient'),
                    Competency(name='SQL', description='Proficient'),
                ],
            ),
        ),
        is_reference=True,
    )

    id2 = add_element_to_database(
        Example(
            abstract='bears like to eat honey',
            profile=Profile(
                domain='Nature',
                competencies=[
                    Competency(name='Honey', description='Loves it'),
                    Competency(name='Bears', description='Likes it'),
                ],
            ),
        ),
        is_reference=False,
    )

    retriever = get_database_as_retriever(max_number_to_retrieve=1, return_type=Example)

    res = retriever.invoke('bear')

    print("Best match in DB for 'bear': ", res)

    db.delete([id1, id2])

    content = db.get()
    for id, doc, metadata in zip(content['ids'], content['documents'], content['metadatas']):
        print(f'ID: {id}')
        print(f'Document: {doc}')
        print(f'Metadata: {metadata}')
        print('---' * 30)
