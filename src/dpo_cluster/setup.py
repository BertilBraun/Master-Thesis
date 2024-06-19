# If the model has already been initialized, we should just exit directly with a non-zero exit code and show an error message

# Setup initial Model 'current-finetuned-model'

# Initial Fine-tuning? - Probably not

# Setup the initial list of (abstracts, best_profile_from_original_model, best_profile_from_last_model) to evaluate the model on
# - Fetch a random set of (~50) authors with at least 4 papers
# - Extract the profiles using the current model and saving both of these in the best_profiles list
