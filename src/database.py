from collections import Counter
import src.defines  # noqa # sets the OpenAI API key and base URL to the environment variables

import os

from enum import Enum
from typing import Type

import chromadb
import chromadb.utils.embedding_functions
from chromadb.api.types import Document, Embedding

from src.types import (
    Combination,
    Example,
    HumanExampleMessage,
    AIExampleMessage,
    Profile,
    Ranking,
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


client = chromadb.PersistentClient(path='data')


class DBEntryType(Enum):
    EXAMPLE = ('example', Example)
    SUMMARY = ('summary', Summary)  # NOTE currently not used
    COMBINATION = ('combination', Combination)
    RANKING = ('ranking', Ranking)


COLLECTIONS = {
    element.value[1]: client.get_or_create_collection(
        name=element.value[0],
        metadata={'reference': 'bool', 'hnsw:space': 'cosine'},
        # embedding_function=chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
        #     api_base=src.openai_defines.BASE_URL_EMBEDDINGS, api_key=src.openai_defines.OPENAI_API_KEY
        # ),
    )
    for element in DBEntryType.__members__.values()
}


def _append_to_database_log(msg: str) -> None:
    with open(DB_LOG_FILE, 'a') as f:
        f.write(msg)


def add_element_to_database(element: DatabaseTypes, is_reference: bool) -> str:
    # Adds the element to the database and returns the ID of the added document. Throws an error if the element was already added.
    log(f'Adding element to DB: {element} as reference: {is_reference}', level=LogLevel.DEBUG)

    id = str(element)
    document = str(element)
    metadata = {'reference': is_reference}

    COLLECTIONS[type(element)].add(
        ids=[id],
        documents=[document],
        metadatas=[metadata],
    )

    _append_to_database_log(f'Added document to DB:\nPage content: {document}\nMetadata: {metadata}')

    return id


def database_size(type: Type[DatabaseTypes]) -> int:
    # Returns the number of elements of the given type in the database
    return COLLECTIONS[type].count()


def _cosine_similarity(a: Embedding, b: Embedding) -> float:
    len_a = sum(x**2 for x in a) ** 0.5
    len_b = sum(x**2 for x in b) ** 0.5
    dot_product = sum(x * y for x, y in zip(a, b))
    return dot_product / (len_a * len_b)


def _text_similarity(a: str, b: str) -> float:
    # count the words in a that are in b including the number of times they appear -> all words of a are in b -> 1
    c = Counter(a.split())
    d = Counter(b.split())
    return sum((c & d).values()) / sum(c.values())


# TODO adjust the threshold
def _greedy_filter(
    query: str,
    documents: list[Document],
    embeddings: list[Embedding],
    number_of_documents: int,
    threshold: float,
) -> list[str]:
    final_documents: list[str] = []
    added_embeddings: list[Embedding] = []
    not_added_documents: list[str] = []

    for document, embedding in zip(documents, embeddings):
        if _text_similarity(query, document) > 0.95:
            # Never add the query itself
            log(
                f'Found query in document - this most likely means the query was already added to the database: {document}',
                level=LogLevel.WARNING,
            )
            continue
        for selected in added_embeddings:
            if _cosine_similarity(embedding, selected) > threshold:
                not_added_documents.append(document)
                break
        else:
            # None of the selected embeddings were too close to the current one
            final_documents.append(document)
            added_embeddings.append(embedding)

            if len(final_documents) == number_of_documents:
                break

    while len(final_documents) < number_of_documents and not_added_documents:
        final_documents.append(not_added_documents.pop(0))

    return final_documents


def get_retriever_getter(max_number_to_retrieve: int) -> RetrieverGetter:
    class DatabaseRetriever(Retriever[DatabaseTypes]):
        def __init__(self, return_type: Type[DatabaseTypes]):
            self.return_type = return_type

        def invoke(self, query: str) -> list[DatabaseTypes]:
            return self.batch([query])[0]

        @timeit('RAG Retrieval', level=LogLevel.DEBUG)
        def batch(self, queries: list[str]) -> list[list[DatabaseTypes]]:
            if max_number_to_retrieve == 0:
                # No retrievals needed - chromadb will crash if we try to retrieve 0 documents
                return [[] for _ in queries]

            log(f'Invoking retriever for {queries=}', level=LogLevel.DEBUG)

            res = COLLECTIONS[self.return_type].query(
                query_texts=queries,
                # We want to get a few more than the maximum number of results we want to return to be able to filter out ones that are too close to each other
                n_results=max_number_to_retrieve * 2,
                where={'reference': True},
                include=['documents', 'embeddings'],
            )

            log(f'Retrieved the following documents: {res}', level=LogLevel.DEBUG)

            assert res['documents'] is not None, f'No documents found for queries: {queries}'
            assert res['embeddings'] is not None, f'No embeddings found for queries: {queries}'

            return [
                [
                    self.return_type.parse(doc)
                    for doc in _greedy_filter(
                        query,
                        similar_docs,
                        similar_embeddings,
                        number_of_documents=max_number_to_retrieve,
                        threshold=0.8,
                    )
                ]
                for query, similar_docs, similar_embeddings in zip(queries, res['documents'], res['embeddings'])
            ]

    def get_retriever(return_type: Type[DatabaseTypes]) -> Retriever[DatabaseTypes]:
        return DatabaseRetriever[DatabaseTypes](return_type=return_type)

    return get_retriever


def get_sample_from_database(type: Type[DatabaseTypes], number_of_samples: int) -> list[DatabaseTypes]:
    # Returns a list of up to number_of_samples elements of the given type from the database
    docs = COLLECTIONS[type].get(limit=number_of_samples, include=['documents'])['documents'] or []
    return [type.parse(doc) for doc in docs]


def get_example_messages(content: str, retriever: Retriever[Example]) -> list[Message]:
    examples = retriever.invoke(content)

    return [
        message
        for i, example in enumerate(examples)
        for message in [
            HumanExampleMessage(content=f'Example {i + 1}:\n{example.abstracts}'),
            AIExampleMessage(content=str(example.profile)),
        ]
    ]


def get_example_messages_json(content: str, retriever: Retriever[Example]) -> list[Message]:
    examples = retriever.invoke(content)

    return [
        message
        for i, example in enumerate(examples)
        for message in [
            HumanExampleMessage(content=f'Example {i + 1}:\n{example.abstracts}'),
            AIExampleMessage(content=example.profile.to_json()),
        ]
    ]


def get_summary_messages(content: str, retriever: Retriever[Summary]) -> list[Message]:
    summaries = retriever.invoke(content)

    return [
        message
        for i, summary in enumerate(summaries)
        for message in [
            HumanExampleMessage(content=f'Example {i + 1}:\n{summary.full_text}'),
            AIExampleMessage(content=summary.summary),
        ]
    ]


def get_ranking_messages_json(content: str, retriever: Retriever[Ranking]) -> list[Message]:
    evaluations = retriever.invoke(content)

    return [
        message
        for i, evaluation in enumerate(evaluations)
        for message in [
            HumanExampleMessage(
                content=f"""Example {i + 1}:
{evaluation.paper_text}


Profile 1:
{evaluation.profiles[0]}


Profile 2:
{evaluation.profiles[1]}"""
            ),
            AIExampleMessage(
                content=f"""{{
    "reasoning": "{evaluation.reasoning}",
    "preferred_profile": {evaluation.preferred_profile + 1}
}}"""
            ),
        ]
    ]


def get_combination_messages(content: str, retriever: Retriever[Combination]) -> list[Message]:
    combinations = retriever.invoke(content)

    return [
        message
        for i, combination in enumerate(combinations)
        for message in [
            HumanExampleMessage(
                content=f'Example {i + 1}:\n' + '\n\n'.join(str(profile) for profile in combination.input_profiles)
            ),
            AIExampleMessage(content=str(combination.combined_profile)),
        ]
    ]


def get_combination_messages_json(content: str, retriever: Retriever[Combination]) -> list[Message]:
    combinations = retriever.invoke(content)

    return [
        message
        for i, combination in enumerate(combinations)
        for message in [
            HumanExampleMessage(
                content=f'Example {i + 1}:\n' + '\n\n'.join(str(profile) for profile in combination.input_profiles)
            ),
            AIExampleMessage(content=combination.combined_profile.to_json()),
        ]
    ]


def dump_database():
    for collection in COLLECTIONS.values():
        print(f'------------------- Collection: {collection.name} -------------------')
        content = collection.get()
        assert content['documents'] is not None, 'No documents found in the database'
        assert content['metadatas'] is not None, 'No metadata found in the database'
        for doc, metadata in zip(content['documents'], content['metadatas']):
            # id and document should be the same
            print(f'Document: {doc}')
            print(f'Metadata: {metadata}')
            print('---' * 30)

        print(f'Number of documents {collection.name}:', collection.count())


def clear_collection(return_type: Type[DatabaseTypes]):
    COLLECTIONS[return_type].delete(COLLECTIONS[return_type].get()['ids'])


def clear_database():
    for collection in client.list_collections():
        client.delete_collection(collection.name)


if __name__ == '__main__':
    from src.types import Profile, Example, Competency

    def bear_example():
        id1 = add_element_to_database(
            Example(
                abstracts='harrison worked at kensho',
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
                abstracts='bears like to eat honey',
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

        COLLECTIONS[Example].delete([id1, id2])

    # bear_example()

    dump_database()
    # clear_database()
    # clear_collection(Ranking)

    # print('Database size (Example):', database_size(Example))
