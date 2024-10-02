from src.logic.types import *
from src.scripts.cleanup_preferences import is_enough_text_to_seem_to_be_a_valid_competency
from src.util.json import dump_json_str, load_json

path = R'C:\Users\berti\OneDrive\Docs\Studium\Semester 8\Masterarbeit\Master-Thesis\dpo_output\preferences\TEMPORARY_LOCAL_ONLY_dev-llama-3-large.json'

if __name__ == '__main__':
    preferences: list[Preference] = load_json(path)

    samples = [dump_json_str(sample) for sample in preferences]
    # Print number of samples and number of unique samples - check if there are direct duplicates
    print(len(samples), len(set(samples)))  # 2400 2398 - good enough for a test run

    total_competencies = sum(len(Profile.parse(preference['chosen']).competencies) for preference in preferences)
    total_valid_competencies = sum(
        len(
            [
                competency
                for competency in Profile.parse(preference['chosen']).competencies
                if is_enough_text_to_seem_to_be_a_valid_competency(str(competency))
            ]
        )
        for preference in preferences
    )

    print('Total competencies: ', total_competencies)
    print('Total valid competencies: ', total_valid_competencies)
    print('Percentage of valid competencies: ', total_valid_competencies / total_competencies)

    # max_length_prompt = max([preference['prompt'] for preference in preferences], key=len)
    # print('Max length prompt: ', len(max_length_prompt))
    # print(f'"""{max_length_prompt}"""')
