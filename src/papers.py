import os

from pypdf import PdfReader
from pyalex import Works, Authors

from src.types import Query, Author
from src.util import cache_to_file, download, timeit

KIT_INSTITUTION_ID = 'i102335020'


@timeit('Get Author by Name')
@cache_to_file('author_cache.cache', Author)
def get_author_by_name(name: str) -> Author | None:
    # Get the OpenAlex author by the author's display name
    author = (
        Authors()
        .filter(affiliations={'institution': {'lineage': KIT_INSTITUTION_ID}})
        .search_filter(display_name=name)
        .get()
    )

    if not author:
        return None

    return Author(
        name=author[0]['display_name'],  # type: ignore
        id=author[0]['id'],  # type: ignore
        count=author[0]['works_count'],  # type: ignore
    )


def verify_is_text(text: str, threshold: float = 0.50) -> str | None:
    # Verify that the text is mostly (more than threshold percent) composed of standard characters, otherwise return None

    if not text:  # Handle empty string case
        return None

    standard_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ,.!?;:\'"-()\n\t'
    count_standard = sum(1 for char in text if char in standard_chars)
    non_standard_ratio = 1 - (count_standard / len(text))

    if non_standard_ratio > threshold:
        return None
    return text


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
            return verify_is_text(f.read())

    # Extract the full text from the PDF
    try:
        reader = PdfReader(file_name)
    except Exception as e:
        print(f'Error reading PDF: {e}')
        return None

    full_text = ''
    for page in reader.pages:
        page_text = page.extract_text()

        # Remove the line breaks
        page_text = page_text.replace('-\n', '').replace('\n', ' ')

        if 'References' in page_text:
            # Strip the references and appendix but add the last page up to the references as well
            full_text += page_text[: page_text.index('References')]
            break
        else:
            full_text += page_text

    # Save the full text to a file to avoid re-extraction
    with open(full_text_file_name, 'w') as f:
        f.write(full_text)

    return verify_is_text(full_text)


@timeit('Get Papers by Author')
@cache_to_file('papers_cache.cache', Query)
def get_papers_by_author(name: str, number_of_papers: int = 5) -> Query:
    # Get the top number_of_papers most cited papers by the author with the given name
    author = get_author_by_name(name)
    assert author, f'Author {name} not found'

    papers = (
        Works()
        .filter(author={'id': author.id})
        .filter(has_abstract=True)
        .filter(has_fulltext=True)
        .filter(open_access={'is_oa': True})
        .filter(fulltext_origin='pdf')
        .sort(cited_by_count='desc')
        .get(per_page=number_of_papers * 2)
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
        author=author.name,
    )


@timeit('Get Authors of KIT')
@cache_to_file('authors_cache.cache', Author)
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

    pprint(get_author_by_name('Peter Sanders'))
    pprint(get_author_by_name('Stiefelhagen'))  # Fuzzy matching

    papers_by_author = get_papers_by_author('Peter Sanders')
    print('We got', len(papers_by_author.full_texts), 'papers by Peter Sanders')
    # pprint(get_papers_by_author('Peter Sanders'))
