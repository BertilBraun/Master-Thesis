NUM_SAMPLES_TO_GENERATE = 2000  # TODO less?

# how many samples to we generate with each extraction of TOP_K_TO_SAMPLE?
#      - 4 papers per sample
#      - TOP_K_TO_SAMPLE extracted profiles
#      - TOP_K_TO_SAMPLE profiles in a tournament
#      - TOP_K_TO_SAMPLE - 1 comparisons in a tournament
#      - TOP_K_TO_SAMPLE = 16 -> 32 usable preferences and 15 comparisons
#      - TOP_K_TO_SAMPLE = 8 -> 12 usable preferences and 7 comparisons
# => higher TOP_K_TO_SAMPLE means more usable preferences with comparativly less comparisons
#    but limited by the number of good profiles we can extract with such a high TEMPERATURE

# TODO how long does extracting NUM_SAMPLES_TO_GENERATE samples take?
#      - NUM_SAMPLES_TO_GENERATE samples / 32 preferences = 63 tournaments
#      - 63 tournaments * 15 comparisons = 945 comparisons
#      - 945 comparisons * 30 seconds / NUM_THREADS_EVALUATE = 1.6 hours
#      - 63 extractions * 30 seconds * TOP_K_TO_SAMPLE / NUM_THREADS_GENERATE = 2.8 hours
# TODO how do generating and evaluating compare in time? Do we need more threads for one or the other?

# TODO are the TOP_K_TO_SAMPLE samples different enough?

PAPERS_PER_SAMPLE = 4
TOP_K_TO_SAMPLE = 16
TEMPERATURE = 0.8  # Prefer more diverse samples so that all TOP_K are different
NUM_EXAMPLES = 1  # TODO or 0?

NUM_THREADS_GENERATE = 3
NUM_THREADS_EVALUATE = 5

# While we have not generated enough samples
# Fetch a random set of authors with at least PAPERS_PER_SAMPLE papers
# Fetch all abstracts of these papers by the author
# Fetch the best matching NUM_EXAMPLES examples in the RAG dataset
# Add a tuple of (author, abstracts, examples) to the samples to generate list

# NUM_THREADS_GENERATE other threads will be running in parallel to generate the samples
# Each thread will fetch one element from the samples to generate list
# Then will call a LLM pipeline on its dedicated GPU to generate the samples
# This call will be with the following parameters:
# - model: 'current-finetuned-model'
# - prompt: to extract with the abstracts and examples
# - temperature: TEMPERATURE
# - top_k: TOP_K_TO_SAMPLE # Generate the top k (different) extracted profiles
# The generated samples will be added to a list of samples to evaluate

# NUM_THREADS_EVALUATE other threads will be running in parallel to evaluate the samples
# Each thread will on startup create a threadlocal database (threadid + timestamp) to store the evaluation samples
# Each thread will fetch one element from the samples to evaluate list
# Then will call a tournament evaluation on the samples with the largest possible LLM
# The evaluation will be written to the threadlocal database with all the preferences
