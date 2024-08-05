# If the model has already been initialized, we should just exit directly with a non-zero exit code and show an error message

# Setup initial Model 'current-finetuned-model'

# Initial Fine-tuning? - Probably not

# Setup the initial list of (abstracts, best_profile_from_original_model, best_profile_from_last_model) to evaluate the model on
# - Fetch a random set of (~50) authors with at least 4 papers
# - Extract the profiles using the current model and saving both of these in the best_profiles list


import os
import sys

import huggingface_hub

from src.database import get_retriever_getter
from src.papers import get_random_english_authors_abstracts
from src.extraction_custom import prompt_for_extract_from_abstracts_custom
from src.types import Example, Profile
from src.dpo_cluster.defines import *
from src.util import json_dumper


if __name__ == '__main__':
    # if MODEL_NAME model exists, exit with non-zero exit code
    if os.path.exists(CURRENT_MODEL_PATH):
        print(f'{CURRENT_MODEL_PATH} already exists. Exiting...')
        sys.exit(1)

    huggingface_hub.login(new_session=False)

    model = get_model(BASE_MODEL_ID)

    # Save the initial model as MODEL_NAME
    model.save_pretrained(CURRENT_MODEL_PATH)
    tokenizer = get_tokenizer()

    with json_dumper(SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE) as dumper:
        for query in get_random_english_authors_abstracts(
            NUMBER_OF_SAMPLES_TO_EVALUATE_THE_IMPROVEMENT_ON_AFTER_TRAINING, PAPERS_PER_SAMPLE
        ):
            abstracts = '\n\n'.join(query.abstracts)

            examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Example).invoke(abstracts)

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

            try:
                profile = Profile.parse('Domain: "' + response)

                dumper(
                    SampleForFineTuningImprovementEvaluation(
                        prompt=prompt,
                        abstracts=query.abstracts,
                        best_profile_from_original_model=str(profile),
                        best_profile_from_last_model=str(profile),
                    )
                )
            except Exception:
                print(f'Error while parsing profile: {query.author}')
                print(f'Prompt: {prompt}')
                print(f'Response: {response}')
