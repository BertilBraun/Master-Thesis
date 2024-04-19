from dataclasses import dataclass
from functools import cache
import os
from pyalex import Works, Authors
from pypdf import PdfReader

from src.types import Query
from src.util import download, timeit

KIT_INSTITUTION_ID = 'i102335020'


@dataclass(frozen=True)
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


@timeit('Load Paper Full Text')
def load_paper_full_text(paper_oa_url: str) -> str | None:
    # Load the full text of a paper from the given Open Access URL

    # Download the paper
    success, file_name = download(paper_oa_url, extension='.pdf')

    if not success:
        # Failed to download the paper -> Cannot extract the full text
        return None

    full_text_file_name = file_name.replace('.pdf', '.txt')
    if os.path.exists(full_text_file_name):
        # The full text has already been extracted and saved to a file
        with open(full_text_file_name, 'r') as f:
            return f.read()

    # Extract the full text from the PDF
    try:
        reader = PdfReader(file_name)
    except Exception as e:
        print(f'Error reading PDF: {e}')
        return None
    texts = [page.extract_text() for page in reader.pages]

    # Strip the references and appendix
    for i, page_text in enumerate(texts):
        if 'References' in page_text:
            texts = texts[:i]
            # add the last page up to the references as well
            texts.append(page_text[: page_text.index('References')])
            break

    # Remove the line breaks
    texts = [text.replace('-\n', '').replace('\n', ' ') for text in texts]

    # Return the full text
    full_text = '\n'.join(texts)

    # Save the full text to a file to avoid re-extraction
    with open(full_text_file_name, 'w') as f:
        f.write(full_text)

    return full_text


@timeit('Get Papers by Author')
@cache
def get_papers_by_author(name: str, number_of_papers: int = 5) -> Query:
    # Get the top number_of_papers most cited papers by the author with the given name

    papers = (
        Works()
        .filter(author={'id': get_author_id_by_name(name)})
        .filter(has_abstract=True)
        .filter(has_fulltext=True)
        .filter(fulltext_origin='pdf')
        .sort(cited_by_count='desc')
        .get(per_page=number_of_papers * 5)
    )

    full_texts: list[str] = []
    abstracts: list[str] = []
    titles: list[str] = []

    for paper in papers:
        if len(full_texts) >= number_of_papers:
            break

        full_text = load_paper_full_text(paper['open_access']['oa_url'])  # type: ignore
        if not full_text:
            continue

        full_texts.append(full_text)
        abstracts.append(paper['abstract'])  # type: ignore
        titles.append(paper['title'])  # type: ignore

    return Query(
        abstracts=abstracts,
        full_texts=full_texts,
        titles=titles,
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

    pprint(get_authors_of_kit())

    print(get_author_id_by_name('Peter Sanders'))

    pprint(get_papers_by_author('Peter Sanders'))
