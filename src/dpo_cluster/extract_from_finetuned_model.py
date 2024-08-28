from tqdm import tqdm


from src.extraction_custom import prompt_for_extract_from_abstracts_custom
from src.dpo_cluster.defines import generate, get_model, get_tokenizer, CURRENT_MODEL_PATH, prompt_messages_to_str
from src.util import json_dumper, log_all_exceptions
from src.database import get_retriever_getter
from src.types import Example, ExtractedProfile, Profile
from src.__main__ import get_queries_from_evaluation_folder


NUM_EXAMPLES = 1


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

        with log_all_exceptions(f'Failed to extract profile from response {author=}: {response=}'):
            profile = ExtractedProfile(
                profile=Profile.parse('Domain: "' + response),
                model='Fine-tuned-Model',
                number_of_examples=NUM_EXAMPLES,
                extraction_function='finetuning',
                extraction_time=0,
            )

            with json_dumper(f'finetuned_profile_{author}.json') as dumper:
                dumper(profile)


if __name__ == '__main__':
    evaluate_authors()
