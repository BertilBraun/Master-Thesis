import lmql

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
        "{i+1}. [THING]" where STOPS_AT(THING, "\n") and len(TOKENS(THING)) > 3 and len(TOKENS(THING)) < 20
        things.append(THING)

    return things
    """


# call your LMQL function like any other Python function
print('Calling LMQL!')

result = packing_list()

print('Done calling LMQL!')

print(result)

# If you dont return anything, then the return value will have the following attributes:
# print('Result:', result.prompt)
# print('Variables:', result.variables)
# print('Distribution:', result.distribution_values, result.distribution_variable)
