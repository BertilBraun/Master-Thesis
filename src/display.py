import os
import json

from enum import Enum
from dataclasses import is_dataclass

from src.log import date_str
from src.types import AuthorExtractionResult, Message


def custom_asdict(obj):
    if is_dataclass(obj):
        result = {}
        for field_name, field_type in obj.__dataclass_fields__.items():
            value = getattr(obj, field_name)
            result[field_name] = custom_asdict(value)
        return result
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, list):
        return [custom_asdict(item) for item in obj]
    elif callable(obj):
        return obj.__qualname__  # Save the function's qualname if it's a callable
    else:
        return obj


def generate_html_file_for_extraction_result(author_result: AuthorExtractionResult):
    json_data = json.dumps(custom_asdict(author_result), indent=4)
    with open('src/template_extraction_result.html', 'r') as file:
        html_template = file.read()

    html_content = html_template.replace('"{{authorData}}"', json_data)

    _write_and_display(html_content, 'extraction_result')


def generate_html_file_for_chat(messages: list[Message], chat_name: str = 'chat'):
    json_data = json.dumps([message.to_dict() for message in messages], indent=4)
    with open('src/template_chat.html', 'r') as file:
        html_template = file.read()

    html_content = html_template.replace('"{{chatData}}"', json_data)

    _write_and_display(html_content, chat_name)


def _write_and_display(html_content: str, file_name: str):
    output_file_path = os.path.abspath(f'logs/{date_str()}/chat_{file_name}.html')

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as file:
        file.write(html_content)

    print(f'You can open the html chat file of {file_name} here:')
    print('file:///' + output_file_path.replace('\\', '/'))


if __name__ == '__main__':
    from src.types import (
        Competency,
        Profile,
        ExtractedProfile,
        EvaluationResult,
        SystemMessage,
        HumanMessage,
        AIMessage,
        AIExampleMessage,
        HumanExampleMessage,
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
    evaluation_result = [EvaluationResult(extracted_profile, 'High accuracy in predictive modeling', 95)]
    author_result = AuthorExtractionResult(evaluation_result, ['Paper on AI', 'Thesis on ML'], 'John Doe')

    generate_html_file_for_extraction_result(author_result)

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
