  


## Running

With LMQL:

Start LMQL Model Serve:

```bash
lmql serve-model (--dtype 8bit only if a GPU is supported)
```

Run the Script itself:

```bash
python -m src
```

In Python we can call LMQL like this:

```python
import lmql

MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' # Example model used from huggingface

@lmql.query(model=MODEL, is_async=False)
def say(phrase):
    """lmql
    # we can seamlessly use 'phrase' in LMQL
    "Say '{phrase}': [TEST]" where len(TOKENS(TEST)) < 25
    # return the result to the caller
    return TEST
    """ # This is the "Prompt" defined in LMQL format.

print(say('Hello World!'))
```
