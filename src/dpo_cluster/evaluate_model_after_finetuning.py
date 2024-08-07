# We should then also evaluate on 10 samples to see if the model has improved
#   do this by having a list of (abstracts, best_profile_from_original_model, best_profile_from_last_model)
#   and then run the extraction on the abstracts with the current model
#   and compare the results to the best_profile_from_original_model and best_profile_from_last_model
#   and see if the current model has improved and write the results to a file
#   afterwards write the new best_profile_from_last_model to the file

# If the model is preferred less than the last model, we should stop the training and keep the last model
#   this is simply done by comparing the number of wins of the current model to the number of wins of the last model
#   if the current model has less wins, we stop the training and keep the last model
#   this is done by exiting the script with a non zero exit code


# Otherwise the script is then responsible to write the new model to the 'current-finetuned-model' file so that it can be used in the next iteration


import random
from typing import Generator
from torch import cuda
from tqdm import tqdm


from src.log import log, ratio
from src.dpo_cluster.defines import *
from src.dpo_cluster.llm_util import parse_llm_from_sysargs
from src.util import json_dumper, load_json, timeblock
from src.database import get_retriever_getter
from src.evaluation import prompt_for_ranking
from src.types import EvaluationResult_from_invalid_response, Profile, Ranking


if __name__ == '__main__':
    START_DATETIME = get_previous_datetime_str()


LLM = parse_llm_from_sysargs()


def evaluate_is_profile1_preferred(profile1: Profile, profile2: Profile, abstracts: list[str]) -> bool:
    profiles = [profile1, profile2]
    random.shuffle(profiles)

    examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking).invoke('\n\n'.join(abstracts))
    prompt = prompt_for_ranking(profiles[0], profiles[1], examples, abstracts)

    response = LLM.invoke(prompt, stop=['\n\n\n\n'], temperature=0.2)

    evaluation = EvaluationResult_from_invalid_response(response)
    preferred_profile_index = Ranking.parse_preferred_profile_json(evaluation)

    return profiles[preferred_profile_index] == profile1


def get_samples_for_fine_tuning_improvement_evaluation(
    samples: list[SampleForFineTuningImprovementEvaluation],
) -> Generator[SampleForFineTuningImprovementEvaluation, None, None]:
    tokenizer = get_tokenizer()
    model = get_model(load_in_8bit=True)  # Load the currently finetuned model

    for sample in tqdm(samples, desc='Evaluating samples'):
        response = generate(
            tokenizer,
            model,
            sample.prompt,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=650,
        )[0]

        yield sample.with_new_profiles(str(Profile.parse('Domain: "' + response)))

    # free the memory again
    del model
    gc.collect()
    cuda.empty_cache()


def get_number_of_wins_current_model(samples_to_evaluate: list[SampleForFineTuningImprovementEvaluation]) -> int:
    return sum(
        evaluate_is_profile1_preferred(
            Profile.parse(sample.best_profile_from_last_model),
            Profile.parse(sample.best_profile_from_original_model),
            sample.abstracts,
        )
        for sample in tqdm(samples_to_evaluate, desc='Evaluating samples')
    )


def evaluate_model() -> None:
    # Load the samples to evaluate before creating the json_dumper, as the dumper will overwrite the file
    old_samples = load_json(
        SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE, SampleForFineTuningImprovementEvaluation
    )
    with timeblock('Evaluating the model after fine-tuning'):
        with json_dumper(SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE) as dumper:
            for sample in get_samples_for_fine_tuning_improvement_evaluation(old_samples):
                dumper(sample)


def compare_models() -> bool:
    # TODO remove
    SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE = R'C:\Users\berti\OneDrive\Docs\Studium\Semester 8\Masterarbeit\Master-Thesis\dpo_output\samples_for_fine_tuning_improvement_evaluation_after_run_2.json'

    with timeblock('Comparing the current model to the baseline model'):
        # reload the samples with the new profiles
        new_samples = load_json(
            SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE, SampleForFineTuningImprovementEvaluation
        )
        number_of_wins_current_model = get_number_of_wins_current_model(new_samples)

    total_samples = len(new_samples)
    log(f'The current model won {ratio(number_of_wins_current_model, total_samples)}.')

    # Return whether the current model is preferred more than the last model
    return number_of_wins_current_model > total_samples * 0.5


if __name__ == '__main__':
    if cuda.is_available():
        evaluate_model()
    else:
        if not compare_models():
            print('The current model is preferred less than the last model. Should keep the last model.')
