import gc
import os
import time
from torch import cuda
from tqdm import tqdm

from src.finetuning.logic.model import generate, get_model, prompt_messages_to_str
from src.finetuning.logic.tokenizer import get_tokenizer
from src.logic.papers import extract_text_from_pdf
from src.extraction.extraction_custom import prompt_for_extract_from_abstracts_custom
from src.finetuning.defines import CURRENT_MODEL_PATH
from src.util import dump_json, load_json, log_all_exceptions
from src.logic.database import get_retriever_getter
from src.logic.types import Example, ExtractedProfile, Profile, Query


NUM_EXAMPLES = 1


def get_queries_from_evaluation_folder(base_folder: str = 'evaluation') -> tuple[dict[str, Query], dict[str, str]]:
    queries: dict[str, Query] = {}
    emails: dict[str, str] = {}

    for folder in os.listdir(base_folder):
        if not os.path.isdir(os.path.join(base_folder, folder)) or 'TODO' in folder or 'DONE' in folder:
            continue

        data = load_json(os.path.join(base_folder, folder, 'data.json'))
        name = data['name']
        emails[name] = data['email']

        abstracts = []
        full_texts = []

        for i in range(1, 6):
            pdf_path = os.path.join(base_folder, folder, f'paper{i}.pdf')
            full_text_path = os.path.join(base_folder, folder, f'paper{i}.txt')

            if os.path.exists(pdf_path):
                # Extract text from PDF
                full_texts.append(extract_text_from_pdf(pdf_path))
            elif os.path.exists(full_text_path):
                with open(full_text_path) as f:
                    full_texts.append(f.read())
            else:
                raise Exception(f'No full text found for {name} paper {i}')

            with open(os.path.join(base_folder, folder, f'paper{i}.abstract.txt')) as f:
                abstracts.append(f.read())

        abstracts = [a for a in abstracts if a]
        full_texts = [t for t in full_texts if t]
        titles = [t for t in data['titles'] if t]

        queries[name] = Query(
            author=name,
            titles=titles,
            abstracts=abstracts,
            full_texts=full_texts,
        )

        # assert all(abstracts), f'Empty abstracts found for author {name}'
        # assert all(full_texts), f'Empty full texts found for author {name}'
        # assert all(data['titles']), f'Empty titles found for author {name}'
        #
        # assert len(queries[name].abstracts) == 5, f'Not enough abstracts found for author {name}'
        # assert len(queries[name].full_texts) == 5, f'Not enough full texts found for author {name}'
        # assert len(queries[name].titles) == 5, f'Not enough titles found for author {name}'

    return queries, emails


def evaluate_authors() -> None:
    tokenizer = get_tokenizer()
    model = get_model(
        CURRENT_MODEL_PATH + '_run_3',
        load_in_8bit=True,
    )  # Load the currently finetuned model

    queries, mails = get_queries_from_evaluation_folder()

    for author, query in tqdm(queries.items(), desc='Evaluating authors'):
        examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Example).invoke(
            '\n\n'.join(query.abstracts)
        )

        prompt_messages = prompt_for_extract_from_abstracts_custom(query.abstracts, examples)

        prompt = prompt_messages_to_str(tokenizer, prompt_messages)
        prompt += '\n<|assistant|>\nDomain: "'

        response = generate(
            tokenizer,
            model,
            prompt,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=650,
        )[0]

        time.sleep(10)  # Sleep to avoid CUDA OOM errors

        gc.collect()
        cuda.empty_cache()

        with log_all_exceptions(f'Failed to extract profile from response {author=}: {response=}'):
            profile = ExtractedProfile(
                profile=Profile.parse('Domain: "' + response),
                model='Fine-tuned-Model',
                number_of_examples=NUM_EXAMPLES,
                extraction_function='finetuning',
                extraction_time=0,
            )

            dump_json(profile, f'finetuned_profile_{author}.json')


if __name__ == '__main__':
    evaluate_authors()
