from dataclasses import dataclass
from functools import cache
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
@cache
def get_author_id_by_name(name: str) -> str:
    # Get the OpenAlex author ID by the author's display name
    return (
        Authors()
        .filter(affiliations={'institution': {'lineage': KIT_INSTITUTION_ID}})
        .search_filter(display_name=name)
        .get()[0]['id']
    )  # type: ignore


def load_paper_full_text(paper_id: str) -> str:
    # TODO strip references and appendix
    return Works().get(paper_id)['full_text']  # type: ignore


@timeit('Get Papers by Author')
@cache
def get_papers_by_author(name: str, number_of_papers: int = 5) -> Query:
    # Get the top number_of_papers most cited papers by the author with the given name

    papers = (
        Works()
        .filter(author={'id': get_author_id_by_name(name)})
        .filter(has_abstract=True)
        .sort(cited_by_count='desc')
        .get(per_page=number_of_papers)
    )

    return Query(
        abstracts=[paper['abstract'] for paper in papers],  # type: ignore
        full_texts=[paper['full_text'] for paper in papers],  # TODO strip references and appendix  # type: ignore
        titles=[paper['title'] for paper in papers],  # type: ignore
        author=name,
    )


@timeit('Get Authors of KIT')
@cache
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
