import src.defines
from src.dpo.dpo_database import DPODatabase, EvaluationType
from src.dpo.jsonbin import JsonBin
from src.types import AuthorResult, Profile, Competency
from src.papers import get_paper_by_title
from src.log import log


def dpo_prompt(abstracts: list[str]) -> str:
    abstracts_str = '\n\n'.join(abstract.strip() for abstract in abstracts)

    return f"""You are a helpful research assistant tasked with analyzing scientific abstracts to extract professional competencies. For these abstracts, identify the primary domain of expertise and list specific competencies demonstrated by the author. Format your findings as follows:
```
Domain: [Short Domain Description]
Competencies:
- [Competency 1]: [Brief description of how Competency 1 is demonstrated across the abstracts]
- [Competency 2]: [Brief description of how Competency 2 is demonstrated across the abstracts]
...
```

Please analyze these scientific abstracts and extract a single professional profile that reflects the competencies and domain of expertise demonstrated throughout. Extract 3 to 8 competencies for these abstracts, providing a clear and concise description for each. The domain description should be a brief label, summarizing the overall area of expertise. Your analysis should be neutral, accurate, and solely based on the content of the abstracts provided. Consider the entire set of abstracts as one cohesive source for a comprehensive competency overview.

Abstracts:

{abstracts_str}"""


def add_to_dataset_from_expert_evaluation(db: DPODatabase) -> None:
    jsonbin = JsonBin(api_key=src.defines.JSONBIN_API_KEY)
    bins = jsonbin.bins()
    print('Found', len(bins), 'bins', bins)

    for bin_id in bins:
        if db.check_existence_by_external_id(bin_id):
            print('Bin', bin_id, 'already exists in the database')
            continue

        bin_data = jsonbin.bin(bin_id)

        author: str = bin_data['author']
        titles: list[str] = bin_data['titles']
        print('Processing bin', bin_id, 'by', author, 'with titles', titles)

        papers = [get_paper_by_title(title) for title in titles]
        abstracts = [paper.abstracts[0] for paper in papers if paper]

        prompt = dpo_prompt(abstracts)

        profile_mapping = {
            int(key): Profile(
                domain=value['profile']['domain'],
                competencies=[
                    Competency(name=comp['name'], description=comp['description'])
                    for comp in value['profile']['competencies']
                ],
            )
            for key, value in bin_data['profiles'].items()
        }

        for preference in bin_data['preferences']:
            instances: list[int] = preference['instances']
            preferred_instance_index: int = preference['preferred_instance']

            if (
                len(instances) != 2
                or preferred_instance_index not in [0, 1]
                or any(instance not in profile_mapping for instance in instances)
            ):
                print('Invalid preference:', preference)
                continue

            preferred_profile = profile_mapping[instances[preferred_instance_index]]
            other_profile = profile_mapping[instances[1 - preferred_instance_index]]

            db.add_entry(prompt, str(preferred_profile), str(other_profile), EvaluationType.EXPERT, author, bin_id)


def add_to_dataset_from_automatic_evaluation(db: DPODatabase, evaluation: AuthorResult) -> None:
    # The assumption is, that all the determined ranking results are in the list preferences

    if db.check_existence_by_author_name_and_eval_type(evaluation.author, EvaluationType.AUTOMATIC):
        print('Automatic evaluation by', evaluation.author, 'already exists in the database')
        return

    papers = [get_paper_by_title(title) for title in evaluation.titles]
    abstracts = [paper.abstracts[0] for paper in papers if paper]

    prompt = dpo_prompt(abstracts)

    for preference in evaluation.preferences:
        preferred_profile = preference.winner.profile
        other_profile = preference.loser.profile

        db.add_entry(prompt, str(preferred_profile), str(other_profile), EvaluationType.AUTOMATIC, evaluation.author)


if __name__ == '__main__':
    db = DPODatabase('dpo.db')

    add_to_dataset_from_expert_evaluation(db)
    # get_dataset_from_automatic_evaluation(db, ...)

    log('Expert data:')
    log(db.get_entries_by_type(EvaluationType.EXPERT), use_pprint=True)
    log('Automatic data:')
    log(db.get_entries_by_type(EvaluationType.AUTOMATIC), use_pprint=True)
