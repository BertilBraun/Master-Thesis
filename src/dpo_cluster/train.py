# This script runs single-threaded for now

# We start by combining the threadlocal evaluation databases into a single Dataset which we can use in training
# We only keep the references to the Databases and load the entries on demand

# The script is then responsible to train the current model 'current-finetuned-model' on the Dataset
# - apparently 3+ epochs are possible (https://x.com/rm_rafailov/status/1751738917613912086)

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
