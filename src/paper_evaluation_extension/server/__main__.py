# A simple flask server that serves two endpoints:
# 1. /get_papers: Given a author name, returns a list of {paper_name, abstract} for all papers that the author has written.
# 2. /process_papers: Given a list of 5-10 {paper_name, abstract}, returns a list of profiles extracted from the papers with different models in this specific format:
# {
#     "model_name": "model_name",
#     "profile": {
#         "domain": "domain",
#         "competencies": [
#             {
#                 "name": "competency_name",
#                 "description": "competency_description"
#             },
#             ...
#         ],
#     }
# }
# 3. /index: Returns the index.html file that is used to interact with the server.
# The server is started by running `python -m src.paper_evaluation_extension.server` from the root directory of the project.

from flask import Flask, request, jsonify

# TODO no relative imports in the main module
import src.defines
from src.logic.database import get_retriever_getter
from src.logic.language_model import OpenAILanguageModel
from src.extraction.extraction_custom import extract_from_abstracts_custom
from src.logic.papers import get_papers_by_author_cached
from src.logic.types.instance_type import Query
from src.util.exceptions import log_all_exceptions
from src.util.json import custom_asdict


app = Flask(__name__)


@app.route('/get_papers', methods=['POST'])
def get_papers():
    assert request.json is not None
    assert 'author_name' in request.json

    author_name = request.json['author_name']
    query = get_papers_by_author_cached(
        author_name,
        number_of_papers=50,
        load_full_text=False,
    )
    papers = [
        {
            'title': title,
            'abstract': abstract,
        }
        for title, abstract in zip(query.titles, query.abstracts)
    ]
    return jsonify(papers)


@app.route('/process_papers', methods=['POST'])
def process_papers():
    assert request.json is not None
    assert 'author_name' in request.json
    assert isinstance(request.json['author_name'], str)
    assert 'papers' in request.json
    assert isinstance(request.json['papers'], list)
    assert all('title' in paper and 'abstract' in paper for paper in request.json['papers'])
    assert 5 <= len(request.json['papers']) <= 10

    author_name = request.json['author_name']
    papers = request.json['papers']
    # sort papers by title to ensure consistent results
    papers = sorted(papers, key=lambda paper: paper['title'])

    profiles = []

    query = Query(
        full_texts=[],
        titles=[paper['title'] for paper in papers],
        abstracts=[paper['abstract'] for paper in papers],
        author=author_name,
    )
    for model in ['gpt-4o', 'gpt-4', 'gpt-3.5-turbo']:  # TODO all in parallel
        retriever_getter = get_retriever_getter(max_number_to_retrieve=2)

        llm = OpenAILanguageModel(
            model,
            base_url=src.defines.BASE_URL_LLM,
            debug_context_name=author_name,
        )

        profile = None
        with log_all_exceptions(f'Error processing {model=}'):
            profile = extract_from_abstracts_custom(query, retriever_getter, llm)

        if profile is not None:
            profiles.append(
                {
                    'model_name': model,
                    'profile': custom_asdict(profile),
                }
            )

    return jsonify(profiles)


@app.route('/index', methods=['GET'])
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run()
