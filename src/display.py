import json
from enum import Enum

from dataclasses import is_dataclass
import os
from src.types import AuthorExtractionResult


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


def generate_html_file(author_result: AuthorExtractionResult):
    json_data = json.dumps(custom_asdict(author_result), indent=4)
    with open('src/template.html', 'r') as file:
        html_template = file.read()

    html_content = html_template.replace('"{{authorData}}"', json_data)

    os.makedirs('results', exist_ok=True)
    with open(f'results/{author_result.author}.html', 'w') as file:
        file.write(html_content)


if __name__ == '__main__':
    from src.types import ExampleType, Instance, Competency, Profile, ExtractedProfile, EvaluationResult

    competencies = [
        Competency('AI', 'Study of artificial intelligence'),
        Competency('ML', 'Machine learning algorithms'),
    ]
    profile = Profile('Technology', competencies)
    example_instance = Instance('OpenAI GPT-4', 150, ExampleType.POSITIVE, lambda x, y, z: profile)
    extracted_profile = ExtractedProfile(profile, example_instance)
    evaluation_result = [EvaluationResult(extracted_profile, 'High accuracy in predictive modeling', 95)]
    author_result = AuthorExtractionResult(evaluation_result, ['Paper on AI', 'Thesis on ML'], 'John Doe')

    generate_html_file(author_result)
