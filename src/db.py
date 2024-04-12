import os
import marqo

from src.types import Example
from src.util import timeit

mq = marqo.Client(url='http://localhost:8882')

INDEX_NAME = 'index'


def does_index_exist(index_name: str) -> bool:
    results = mq.get_indexes()['results']
    return any(index['indexName'] == index_name for index in results)


def create_index(index_name: str) -> None:
    print('Creating index')
    mq.create_index(index_name)


if not does_index_exist(INDEX_NAME):
    create_index(INDEX_NAME)


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
        rating: int | None = None,
        is_reference: bool = False,
    ) -> None:
        mq.index(INDEX_NAME).add_documents(
            [
                {
                    'text': str(example),
                    'author': author,
                    'rating': rating,
                    'reference': is_reference,
                }
            ],
            tensor_fields=['text'],
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
    def search(query: str | dict[str, float], limit: int = 10):
        # Return the top `limit` results which match the query as best as possible
        results = mq.index(INDEX_NAME).search(
            q=query,
            limit=limit,
            show_highlights=False,
            filter_string='reference:true',  # 'reference:false AND rating:[60 TO 100]',
            # searchable_attributes=['text'],
        )

        return [Example.parse(result['text']) for result in results['hits']]

    @staticmethod
    def search_negative(query: str, limit: int = 10):
        # Return the top `limit` results which match the query as negatively as possible
        return DB.search({query: -1.0}, limit)

    @staticmethod
    @timeit('DB.update')
    def update(example: Example, rating: int, is_reference: bool) -> None:
        # Update the rating and reference status of the example
        hit = mq.index(INDEX_NAME).search(
            str(example),
            limit=1,
            search_method=marqo.SearchMethods.LEXICAL,
        )['hits'][0]

        mq.index(INDEX_NAME).update_documents(
            [
                {
                    '_id': hit['id'],
                    'rating': rating,
                    'reference': is_reference,
                }
            ]
        )

        _append_to_log(
            f"""Updated example in DB:
Example: {example}
Rating: {rating}
Reference: {is_reference}
"""
        )

    @staticmethod
    def clear():
        # Clear the database
        mq.index(INDEX_NAME).delete()
        create_index(INDEX_NAME)

        _append_to_log('Cleared DB\n')
