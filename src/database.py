import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from src.util import timeit
from src.types import Example


db = Chroma(
    persist_directory='data',
    collection_name='example',
    embedding_function=OpenAIEmbeddings(),
    collection_metadata={'author': 'string', 'rating': 'int', 'reference': 'bool'},
)


def _append_to_log(msg: str) -> None:
    os.makedirs('logs', exist_ok=True)
    with open('logs/db.log', 'a') as f:
        f.write(msg)


class DB:
    @staticmethod
    @timeit('DB.add')
    def add(
        example: Example,
        author: str,
        rating: int = -1,
        is_reference: bool = False,
    ) -> None:
        db.add_documents(
            [
                Document(
                    page_content=str(example),
                    metadata={
                        'author': author,
                        'rating': rating,
                        'reference': is_reference,
                    },
                ),
            ],
        )

        _append_to_log(
            f"""Added example to DB:
Example: {example}
Author: {author}
Rating: {rating}
Reference: {is_reference}
"""
        )

    @staticmethod
    @timeit('DB.search')
    def search(query: str, limit: int = 10) -> list[Example]:
        # Return the top `limit` results which match the query as best as possible
        results = db.similarity_search(
            query,
            k=limit,
            filter={'reference': True},  # type: ignore  # 'reference:false AND rating:[60 TO 100]'
        )

        return [Example.parse(result.page_content) for result in results]

    @staticmethod
    def as_retriever(limit: int) -> Runnable[str, list[Example]]:
        retriever = db.as_retriever(
            search_kwargs={
                'k': limit,
                # 'filter': {'reference': True}})
            }
        ).with_fallbacks([RunnableLambda(lambda query: [])])

        def parse_documents(docs: list[Document]) -> list[Example]:
            return [Example.parse(doc.page_content) for doc in docs]

        return retriever | parse_documents

    @staticmethod
    def clear():
        # Clear the database
        db.delete_collection()
        _append_to_log('Cleared DB\n')


if __name__ == '__main__':
    from src.types import Profile, Example, Competency

    DB.add(
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
        author='harrison',
        rating=5,
        is_reference=True,
    )

    DB.add(
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
        author='bears',
        rating=3,
        is_reference=False,
    )

    retriever = DB.as_retriever(limit=1)

    res = retriever.invoke('bear')

    print(retriever, res)
