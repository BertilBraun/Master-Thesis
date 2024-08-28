from concurrent.futures import ThreadPoolExecutor
import os
from time import sleep

from pypdf import PdfReader
from pyalex import Works, Authors

from src.log import LogLevel, log
from src.types import Query, Author
from src.util import cache_to_file, download, timeit

KIT_INSTITUTION_ID = 'i102335020'


@timeit('Get Author by Name')
def get_author_by_name(name: str, KIT_only: bool) -> Author | None:
    # Get the OpenAlex author by the author's display name
    request = Authors().search_filter(display_name=name)
    if KIT_only:
        request = request.filter(affiliations={'institution': {'id': KIT_INSTITUTION_ID}})
    authors = request.get()

    if not authors:
        return None

    return Author(
        name=authors[0]['display_name'],  # type: ignore
        id=authors[0]['id'],  # type: ignore
        count=authors[0]['works_count'],  # type: ignore
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

    return extract_text_from_pdf(file_name, full_text_file_name)


def extract_text_from_pdf(file_name: str, full_text_file_name: str | None = None) -> str | None:
    # Extract the full text from the PDF
    try:
        reader = PdfReader(file_name)
    except Exception as e:
        log(f'Error reading PDF: {e}', level=LogLevel.WARNING)
        return None

    full_text = ''
    for page in reader.pages:
        page_text = page.extract_text()

        # Clean up the text

        # Remove the line breaks
        page_text = page_text.replace('-\n', '').replace('\n', ' ')

        # Remove multiple spaces
        while '  ' in page_text:
            page_text = page_text.replace('  ', ' ')

        # if references are found case insensitive, break the loop
        if 'references' in page_text.lower():
            # Strip the references and appendix but add the last page up to the references as well
            full_text += page_text[: page_text.lower().index('references')]
            break
        else:
            full_text += page_text

    # Save the full text to a file to avoid re-extraction
    if full_text_file_name:
        with open(full_text_file_name, 'w') as f:
            f.write(full_text)

    return verify_is_text(full_text)


@timeit('Get Paper by Title')
@cache_to_file('paper_cache.json', Query)
def get_paper_by_title(title: str, load_full_text: bool = False) -> Query | None:
    # Get the paper with the given title
    papers = Works().search_filter(title=title).get()

    if not papers:
        return None

    paper = papers[0]

    if load_full_text:
        full_text = load_paper_full_text(paper['open_access']['oa_url'])  # type: ignore
        if not full_text:
            log('Failed to load full text for paper:', title, level=LogLevel.WARNING)
            return None

    return Query(
        abstracts=[paper['abstract']],  # type: ignore
        full_texts=[full_text] if load_full_text else [],
        titles=[paper['title']],  # type: ignore
        author=paper['authorships'][0]['author']['display_name'],  # type: ignore
    )


@cache_to_file('papers_cache.json', Query)
def get_papers_by_author_cached(
    name: str, number_of_papers: int = 5, KIT_only: bool = True, load_full_text: bool = True
) -> Query:
    # Get the top number_of_papers most cited papers by the author with the given name
    author = get_author_by_name(name, KIT_only)
    assert author, f'Author {name} not found'

    return get_papers_by_author(author.name, author.id, number_of_papers, KIT_only, load_full_text)


@timeit('Get Papers by Author')
def get_papers_by_author(
    author_name: str, author_id: str, number_of_papers: int = 5, KIT_only: bool = True, load_full_text: bool = True
) -> Query:
    works = Works().filter(language='en').filter(author={'id': author_id}).filter(has_abstract=True)
    if load_full_text:
        works = works.filter(has_fulltext=True).filter(open_access={'is_oa': True}).filter(fulltext_origin='pdf')
    papers = works.sort(cited_by_count='desc').get(per_page=number_of_papers * 10)

    full_texts: list[str] = []
    abstracts: list[str] = []
    titles: list[str] = []

    for paper in papers:
        if len(full_texts) >= number_of_papers:
            break

        if load_full_text:
            if not paper['open_access']['oa_url']:  # type: ignore
                continue
            full_text = load_paper_full_text(paper['open_access']['oa_url'])  # type: ignore
            if not full_text:
                continue

            full_texts.append(full_text)
        else:
            full_texts.append(paper['abstract'])  # type: ignore
        abstracts.append(paper['abstract'])  # type: ignore
        titles.append(paper['title'])  # type: ignore

    assert len(abstracts) >= number_of_papers, f'Not enough papers found for author {author_name}'

    return Query(
        abstracts=abstracts,
        full_texts=full_texts,
        titles=titles,
        author=author_name,
    )


@timeit('Get Random English Authors Abstracts')
def get_random_english_authors_abstracts(number_of_authors: int, number_of_papers_per_author: int) -> list[Query]:
    authors = (
        Authors()
        .filter(affiliations={'institution': {'country_code': 'US'}})
        .filter(works_count=f'>{number_of_papers_per_author}')
        .sample(number_of_authors * 2)
        .get(per_page=min(number_of_authors * 2, 200))
    )
    sleep(1)

    queries: list[Query] = []

    # in parallel, get the papers for each author
    with ThreadPoolExecutor() as executor:

        def get_papers(author_name: str, author_id: str) -> Query:
            return get_papers_by_author(
                author_name, author_id, number_of_papers_per_author, load_full_text=False, KIT_only=False
            )

        for batch_start in range(0, len(authors), 10):
            batch = authors[batch_start : batch_start + 10]
            futures = [executor.submit(get_papers, author['display_name'], author['id']) for author in batch]  # type: ignore

            queries += [future.result() for future in futures]
            sleep(1)

    queries = [query for query in queries if len(query.abstracts) >= number_of_papers_per_author]

    if len(queries) < number_of_authors:
        queries += get_random_english_authors_abstracts(number_of_authors - len(queries), number_of_papers_per_author)

    return queries


@timeit('Get Authors of KIT')
@cache_to_file('authors_cache.json', Author)
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
    log(get_authors_of_kit(), use_pprint=True)

    log(get_author_by_name('Peter Sanders', KIT_only=True), use_pprint=True)
    log(get_author_by_name('Stiefelhagen', KIT_only=True), use_pprint=True)  # Fuzzy matching

    papers_by_author = get_papers_by_author_cached('Peter Sanders')
    log('We got', len(papers_by_author.full_texts), 'papers by Peter Sanders')
    # log(get_papers_by_author_cached('Peter Sanders'), use_pprint=True)
