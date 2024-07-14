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


from concurrent.futures import Future, ProcessPoolExecutor
import sys
import partialjson
from torch import float16, cuda

from peft import AutoPeftModelForCausalLM

from src.dpo_cluster.train import TRAINING_OUTPUT_DIR
from src.dpo_cluster.defines import *
from src.util import dump_json, load_json
from src.database import get_retriever_getter
from src.evaluation import default_round_evaluator, prompt_for_ranking, run_tournament_ranking
from src.types import EvaluationResult, Profile, Ranking


if __name__ == '__main__':
    START_DATETIME = get_previous_datetime_str()


def evaluate_is_profile1_preferred(model, tokenizer, profile1: Profile, profile2: Profile, abstracts: list[str]) -> int:
    profiles = [profile1, profile2]

    examples = get_retriever_getter(max_number_to_retrieve=NUM_EXAMPLES)(Ranking).invoke('\n\n'.join(abstracts))

    def evaluator(profile_index1: int, profile_index2: int) -> EvaluationResult:
        prompt_messages = prompt_for_ranking(profiles[profile_index1], profiles[profile_index2], examples, abstracts)
        prompt = prompt_messages_to_str(tokenizer, prompt_messages)

        response = generate(
            tokenizer,
            model,
            prompt,
            num_return_sequences=1,
            do_sample=False,
            max_new_tokens=350,
        )[0]

        return partialjson.JSONParser().parse(response)

    tournament = run_tournament_ranking(
        list(range(len(profiles))),
        default_round_evaluator(evaluator),
        do_shuffle=True,
    )

    return tournament.match.winner == 0


def get_samples_for_fine_tuning_improvement_evaluation() -> list[SampleForFineTuningImprovementEvaluation]:
    # Load PEFT model on CPU
    tokenizer = get_tokenizer()
    model = AutoPeftModelForCausalLM.from_pretrained(
        TRAINING_OUTPUT_DIR,
        torch_dtype=float16,
        low_cpu_mem_usage=True,
    )

    samples_for_fine_tuning_improvement_evaluation: list[SampleForFineTuningImprovementEvaluation] = []

    for element in load_json(SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE):
        sample = SampleForFineTuningImprovementEvaluation.from_json(element)

        response = generate(
            tokenizer,
            model,
            sample.prompt,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=650,
        )[0]

        samples_for_fine_tuning_improvement_evaluation.append(
            sample.with_new_profiles(
                str(Profile.parse(response)),
            )
        )

    # free the memory again
    del model
    gc.collect()
    cuda.empty_cache()

    return samples_for_fine_tuning_improvement_evaluation


def evaluate_profile_preference_batch(
    index: int, samples_to_evaluate: list[SampleForFineTuningImprovementEvaluation]
) -> int:
    tokenizer = get_tokenizer(EVALUATION_MODEL_ID)
    model = get_model(
        EVALUATION_MODEL_ID,
        device=f'cuda:{index}',
        load_in_8bit=True,
        use_flash_attention=USE_FLASH_ATTENTION_FOR_EVALUATION,
    )

    number_of_wins_current_model = 0

    for sample in samples_to_evaluate:
        if evaluate_is_profile1_preferred(
            model,
            tokenizer,
            Profile.parse(sample.best_profile_from_last_model),
            Profile.parse(sample.best_profile_from_original_model),
            sample.abstracts,
        ):
            number_of_wins_current_model += 1

    return number_of_wins_current_model


def get_number_of_wins_current_model(samples_to_evaluate: list[SampleForFineTuningImprovementEvaluation]) -> int:
    number_of_wins_current_model = 0

    samples_processed = 0
    samples_per_thread = min(len(samples_to_evaluate) // NUM_THREADS_EVALUATE, 20)

    with ProcessPoolExecutor() as executor:
        while samples_processed < len(samples_to_evaluate):
            eval_futures: list[Future[int]] = []
            for i in range(NUM_THREADS_EVALUATE):
                eval_futures.append(
                    executor.submit(
                        evaluate_profile_preference_batch,
                        i,
                        samples_to_evaluate[samples_processed : samples_processed + samples_per_thread],
                    )
                )
                samples_processed += samples_per_thread

            for future in eval_futures:
                number_of_wins_current_model += future.result()

    return number_of_wins_current_model


def evaluate_model() -> bool:
    samples_for_fine_tuning_improvement_evaluation = get_samples_for_fine_tuning_improvement_evaluation()

    # Make a backup of the old file
    dump_json(
        load_json(SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE),
        SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE + START_DATETIME + '.old',
    )
    dump_json(
        samples_for_fine_tuning_improvement_evaluation,
        SAMPLES_FOR_FINE_TUNING_IMPROVEMENT_EVALUATION_FILE,
    )

    number_of_wins_current_model = get_number_of_wins_current_model(samples_for_fine_tuning_improvement_evaluation)

    # Return whether the current model is preferred more than the last model
    return number_of_wins_current_model > len(samples_for_fine_tuning_improvement_evaluation) * 0.5


if __name__ == '__main__':
    if not evaluate_model():
        print('The current model is preferred less than the last model. Should keep the last model.')
        sys.exit(1)  # exit with non-zero exit code to stop the iteration
