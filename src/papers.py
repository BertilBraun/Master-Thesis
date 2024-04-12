from dataclasses import dataclass
from pyalex import Works, Authors

from src.types import Query
from src.util import timeit

KIT_INSTITUTION_ID = 'i102335020'


@dataclass
class Author:
    name: str
    id: str
    count: int


@timeit('Get Author Id by Name')
def get_author_id_by_name(name: str) -> str:
    # Get the OpenAlex author ID by the author's display name
    return Authors().search_filter(display_name=name).get()[0]['id']  # type: ignore


@timeit('Get Papers by Author')
def get_papers_by_author(name: str) -> list[Query]:
    # Get the top 5 most cited papers by the author with the given name

    papers = (
        Works()
        .filter(author={'id': get_author_id_by_name(name)})
        .filter(has_abstract=True)
        .sort(cited_by_count='desc')
        .get(per_page=5)
    )

    return [
        Query(
            abstract=paper['abstract'],  # type: ignore
            author=name,
        )
        for paper in papers
    ]


def get_authors_of_kit(count: int = 100) -> list[Author]:
    authors = (
        Works()
        .filter(authorships={'institutions': {'lineage': KIT_INSTITUTION_ID}})
        .group_by('author.id')
        .get(per_page=count)
    )

    return [
        Author(
            name=author['key_display_name'],  # type: ignore
            id=author['key'],  # type: ignore
            count=author['count'],  # type: ignore
        )
        for author in authors
    ]


if __name__ == '__main__':
    from pprint import pprint

    print(get_author_id_by_name('Peter Sanders'))

    pprint(get_papers_by_author('Peter Sanders'))

    pprint(get_authors_of_kit())
