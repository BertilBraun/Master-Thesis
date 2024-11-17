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
import requests

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
    for model in [
        'gemma2-9b-it',
        'llama-3.1-70b-versatile',
        'mixtral-8x7b-32768',
        'llama-3.1-8b-instant',
        'llama-3.2-1b-preview',
        'llama-3.2-90b-text-preview',
    ]:  # ['gpt-4o', 'gpt-4', 'gpt-3.5-turbo']:  # TODO all in parallel
        retriever_getter = get_retriever_getter(max_number_to_retrieve=2)

        llm = OpenAILanguageModel(
            model,
            base_url=src.defines.GROQ_BASE_URL,  # src.defines.BASE_URL_LLM,
            api_key=src.defines.GROQ_API_KEY,  # src.defines.API_KEY_LLM,
            debug_context_name=author_name,
        )

        profile = None
        with log_all_exceptions(f'Error processing {model=}'):
            profile = extract_from_abstracts_custom(query, retriever_getter, llm)

        if profile is not None:
            profile.competencies = profile.competencies[:10]
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


@app.route('/upload_profiles', methods=['POST'])
def upload_profiles():
    # Ensure the request contains profiles
    assert request.json is not None
    assert 'author_name' in request.json
    assert isinstance(request.json['author_name'], str)
    assert 'profiles' in request.json
    assert isinstance(request.json['profiles'], list)
    assert 'abstracts' in request.json
    assert isinstance(request.json['abstracts'], list)

    author_name = request.json['author_name']
    profiles = request.json['profiles']
    abstracts = request.json['abstracts']

    url = 'https://api.jsonbin.io/v3/b'
    headers = {
        'Content-Type': 'application/json',
        'X-Master-Key': src.defines.JSONBIN_API_KEY,
        'X-Bin-Name': f'Uploaded Profiles: {author_name}',
        'X-Bin-Private': 'true',
    }

    response = requests.post(
        url,
        json={
            'author': author_name,
            'profiles': profiles,
            'abstracts': abstracts,
        },
        headers=headers,
    )
    response_data = response.json()

    if response.status_code != 200:
        return jsonify({'message': response_data.get('message', 'Unknown error')}), response.status_code

    bin_id = response_data['metadata']['id']
    return jsonify({'binId': bin_id}), 200


if __name__ == '__main__':
    # Run with: python -m src.paper_evaluation_extension.server
    app.run()
