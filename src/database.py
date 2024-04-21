import src.openai_defines  # noqa # sets the OpenAI API key and base URL to the environment variables

import os

from enum import Enum
from typing import Type

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

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
    DatabaseTypes,
)
from src.util import timeit
from src.log import LogLevel, log

DB_LOG_FOLDER = 'logs'
DB_LOG_FILE = DB_LOG_FOLDER + '/db.log'
os.makedirs(DB_LOG_FOLDER, exist_ok=True)

db = Chroma(
    persist_directory='data',
    collection_name='database',
    embedding_function=OpenAIEmbeddings(base_url=src.openai_defines.LOCAL_AI_CODER),
    collection_metadata={'reference': 'bool', 'type': 'string'},
)


class DBEntryType(Enum):
    EXAMPLE = ('example', Example)
    SUMMARY = ('summary', Summary)
    EVALUATION = ('evaluation', Evaluation)
    COMBINATION = ('combination', Combination)


# Maps from Example | Summary | Evaluation | Combination to 'example' | 'summary' | 'evaluation' | 'combination'
CLASS_TO_TYPE = {element.value[1]: element.value[0] for element in DBEntryType.__members__.values()}


def _append_to_database_log(msg: str) -> None:
    with open(DB_LOG_FILE, 'a') as f:
        f.write(msg)


def add_element_to_database(element: DatabaseTypes, is_reference: bool) -> str:
    # Adds the element to the database and returns the ID of the added document

    metadata = {
        'reference': is_reference,
        'type': CLASS_TO_TYPE[type(element)],
    }

    ids = db.add_documents([Document(page_content=str(element), metadata=metadata)])

    _append_to_database_log(f'Added document to DB:\nPage content: {element}\nMetadata: {metadata}')

    return ids[0]


def get_retriever_getter(max_number_to_retrieve: int) -> RetrieverGetter:
    class DatabaseRetriever(Retriever[DatabaseTypes]):
        def __init__(self, return_type: Type[DatabaseTypes]):
            self.return_type = return_type

        @timeit('RAG Retrieval')
        def invoke(self, query: str) -> list[DatabaseTypes]:
            log(f'Invoking retriever for {query=}', level=LogLevel.DEBUG)

            res = db.similarity_search(
                query,
                k=max_number_to_retrieve,
                filter={
                    'type': CLASS_TO_TYPE[self.return_type],
                    'reference': True,  # type: ignore (for some reason the filter is a dict[str, str] instead of dict[str, Any] which is what the metadata actually is)
                },
            )

            log(f'Retrieved the following documents: {res}', level=LogLevel.DEBUG)

            return [self.return_type.parse(doc.page_content) for doc in res]

        def batch(self, queries: list[str]) -> list[list[DatabaseTypes]]:
            return [self.invoke(query) for query in queries]

    def get_retriever(return_type: Type[DatabaseTypes]) -> Retriever[DatabaseTypes]:
        return DatabaseRetriever[DatabaseTypes](return_type=return_type)

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
            HumanExampleMessage(content=evaluation.paper_text + '\n\n' + str(evaluation.profile)),
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
        is_reference=True,
    )

    retriever = get_retriever_getter(max_number_to_retrieve=2)(Example)

    res = retriever.invoke('bears')

    print("Best match in DB for 'bear': ", res)

    db.delete([id1, id2])

    content = db.get()
    for id, doc, metadata in zip(content['ids'], content['documents'], content['metadatas']):
        print(f'ID: {id}')
        print(f'Document: {doc}')
        print(f'Metadata: {metadata}')
        print('---' * 30)
