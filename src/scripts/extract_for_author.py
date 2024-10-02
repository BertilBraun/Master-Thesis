import sys
from src.extraction.process_authors import process_author_list

from src.logic.display import write_author_result


if __name__ == '__main__':
    authors_list = ['Sanders', 'Oberweis', 'Stiefelhagen'] if sys.argv[1] == 'all' else [sys.argv[1]]

    for result in process_author_list(authors_list):
        write_author_result(result, result.author + '@kit.edu')
