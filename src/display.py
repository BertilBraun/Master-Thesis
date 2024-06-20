import os
import json

from src.log import date_str
from src.types import AuthorResult, Message
from src.util import custom_asdict


def generate_html_file_for_extraction_result(author_result: AuthorResult):
    json_data = json.dumps(custom_asdict(author_result), indent=4)
    with open('src/template_extraction_result.html', 'r') as file:
        html_template = file.read()

    html_content = html_template.replace('"{{authorData}}"', json_data)

    output_file_path = os.path.abspath(f'results/{author_result.author}.html')
    _write_and_display(html_content, output_file_path)


def generate_html_file_for_tournament_evaluation(author_result: AuthorResult):
    json_data = json.dumps(custom_asdict(author_result), indent=4)
    with open('src/template_tournament_evaluation.html', 'r') as file:
        html_template = file.read()

    html_content = html_template.replace('"{{authorData}}"', json_data)

    output_file_path = os.path.abspath(f'results/{author_result.author}.evaluation.html')
    _write_and_display(html_content, output_file_path)


def generate_html_file_for_tournament_ranking_result(author_result: AuthorResult):
    json_data = json.dumps(custom_asdict(author_result), indent=4)
    with open('src/template_tournament_ranking_result.html', 'r') as file:
        html_template = file.read()

    html_content = html_template.replace('"{{authorData}}"', json_data)

    output_file_path = os.path.abspath(f'results/{author_result.author}.tournament.html')
    _write_and_display(html_content, output_file_path)


def dump_author_result_to_json(author_result: AuthorResult):
    output_file_path = os.path.abspath(f'results/{author_result.author}.json')
    with open(output_file_path, 'w') as file:
        json.dump(custom_asdict(author_result), file, indent=4)


def generate_html_file_for_chat(messages: list[Message], chat_name: str = 'chat'):
    json_data = json.dumps([message.to_dict() for message in messages], indent=4)
    with open('src/template_chat.html', 'r') as file:
        html_template = file.read()

    html_content = html_template.replace('"{{chatData}}"', json_data).replace('"{{fileName}}"', chat_name)

    output_file_path = os.path.abspath(f'logs/{date_str()}/chat_{chat_name}.html')
    _write_and_display(html_content, output_file_path)


def _write_and_display(html_content: str, output_file_path: str):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as file:
        file.write(html_content)

    print('file:///' + output_file_path.replace('\\', '/'))


if __name__ == '__main__':
    from src.types import (
        Competency,
        Profile,
        ExtractedProfile,
        SystemMessage,
        HumanMessage,
        AIMessage,
        AIExampleMessage,
        HumanExampleMessage,
        TournamentNode,
        RankingResult,
    )

    competencies = [
        Competency('AI', 'Study of artificial intelligence'),
        Competency('ML', 'Machine learning algorithms'),
    ]
    profile = Profile('Technology', competencies)
    extracted_profile = ExtractedProfile(
        profile,
        model='OpenAI GPT-4',
        number_of_examples=2,
        extraction_function='extraction_function',
        extraction_time=0.5,
    )
    ranking_result = RankingResult(profiles=(1, 1), reasoning='Same profile', preferred_profile_index=0)
    author_result = AuthorResult(
        TournamentNode(match=ranking_result, children=[]),
        {1: extracted_profile},
        ['Paper on AI', 'Thesis on ML'],
        'John Doe',
    )

    generate_html_file_for_extraction_result(author_result)

    generate_html_file_for_tournament_evaluation(author_result)

    generate_html_file_for_tournament_ranking_result(author_result)

    messages = [
        SystemMessage('Welcome to the Chat!'),
        HumanExampleMessage('I am a student'),
        AIExampleMessage('Hello student!'),
        HumanMessage('Hello!'),
        AIMessage('Hi, how can I help you today?'),
        HumanMessage('I need help with my homework'),
        AIMessage('Sure! What subject is it?'),
        HumanMessage('It is about AI and ML'),
        AIMessage('Great! I can help you with that!'),
    ]

    generate_html_file_for_chat(messages)
