from collections import Counter
from src.logic.types import *
from src.util.json import dump_json, load_json

PATH = R'C:\Users\berti\OneDrive\Docs\Studium\Semester 8\Masterarbeit\Master-Thesis\dpo_output\preferences\TEMPORARY_LOCAL_ONLY_dev-llama-3-large.json'


def is_enough_text_to_seem_to_be_a_valid_competency(
    competency: str, threshold: float = 0.7, max_gram_percent: float = 0.06
) -> bool:
    # Define valid characters
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    # Calculate percentage of valid characters
    num_chars = sum(1 for char in competency if char in chars)
    percent = num_chars / len(competency)

    # Check if the percentage of valid characters is above the threshold
    if percent <= threshold:
        return False

    # Generate 2-grams and 3-grams
    ngrams = [competency[i : i + n] for n in (3,) for i in range(len(competency) - n + 1)]

    # Count occurrences of each n-gram
    ngram_counts = Counter(ngrams)

    # Calculate the total number of n-grams
    total_ngrams = sum(ngram_counts.values())
    # max_ngram_percent = max(ngram_counts.values()) / total_ngrams
    # if max_ngram_percent > max_gram_percent:
    #     print(max_ngram_percent, ': ', competency)

    # Check if any n-gram exceeds the max_gram_percent threshold
    for count in ngram_counts.values():
        if count / total_ngrams > max_gram_percent:
            return False

    return True


if __name__ == '__main__':
    preferences: list[Preference] = load_json(PATH)

    # Write cleaned preferences to file - remove competencies that are not valid competencies and limit to 8 competencies
    for preference in preferences:
        for key in ('chosen', 'rejected'):
            profile = Profile.parse(preference[key])
            competencies = [
                competency
                for competency in profile.competencies
                if is_enough_text_to_seem_to_be_a_valid_competency(
                    str(competency), max_gram_percent=0.05 if key == 'chosen' else 0.06
                )
            ][:8]  # max 8 competencies
            preference[key] = str(Profile(domain=profile.domain, competencies=competencies))

    dump_json(preferences, PATH + '.clean')
