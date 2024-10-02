from src.extraction.process_authors import process_author_map
from src.logic.display import write_author_result
from src.finetuning.extract_from_finetuned_model import get_queries_from_evaluation_folder


if __name__ == '__main__':
    # for each folder in evaluation, process the author
    # The name and email are stored in the data.json file
    # There should be 5 papers for each author
    # Each paper should have an abstract and a full text
    # The full text should be stored in a files named paper1.txt, paper2.txt, etc. or paper1.pdf, paper2.pdf, etc.
    # The abstract should be stored in a files named paper1.abstract.txt, paper2.abstract.txt, etc.
    # In case the full text is a PDF, the first step is to extract the text from the PDF
    # Then setup a query for each author with the abstracts and full texts
    # Assert that each query has 5 abstracts and 5 full texts

    queries, emails = get_queries_from_evaluation_folder()

    for result in process_author_map(queries):
        write_author_result(result, emails[result.author], f'evaluation/{result.author}')
