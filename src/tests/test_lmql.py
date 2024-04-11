import os
import lmql
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'  # 'meta-llama/Llama-2-7b-chat-hf'  # Example model specification


@lmql.query(model=MODEL, is_async=False)
def say(phrase):
    """lmql
    # we can seamlessly use 'phrase' in LMQL
    "Say '{phrase}': [TEST]" where len(TOKENS(TEST)) < 25
    # return the result to the caller
    return TEST
    """


@lmql.query(model=MODEL, is_async=False)
def packing_list():
    """lmql
    "My packing list for the trip:\n\n"

    things = []
    for i in range(4):
        "{i+1}. [THING]\n" where STOPS_BEFORE(THING, "\n") and len(WORDS(THING)) > 2 and len(WORDS(THING)) < 20
        things.append(THING)

    return things
    """


m: lmql.LLM = lmql.model(MODEL)


def test():
    # call your LMQL function like any other Python function
    print('Calling LMQL!')

    start_time = time.time()

    # result = packing_list()
    result = m.generate_sync(
        "I'm going on a beach vacation and need to pack. List the items I should bring:", max_tokens=100
    )

    print('Done calling LMQL!')
    print('Inference time:', time.time() - start_time)

    print(result)

    # If you dont return anything, then the return value will have the following attributes:
    # print('Result:', result.prompt)
    # print('Variables:', result.variables)
    # print('Distribution:', result.distribution_values, result.distribution_variable)


test()
test()
test()
test()
