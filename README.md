# Masters Thesis

## Running

With LMQL:

Start LMQL Model Serve:

```bash
lmql serve-model (--dtype 8bit - only if a GPU is supported)
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

Reference to all the available LMQL operations can be found [here](https://github.com/eth-sri/lmql/blob/main/src/lmql/ops/ops.py#L917).

LMQL Seems to take twice as long as transformers to generate the output.

Is this fast enough? Any alternatives? Guidance? Langchain with YAML (Idk how much that fucks up the prompt though)?

## Vector Database (Marqo)

Follow: [Marqo Docs](https://github.com/marqo-ai/marqo)

```bash
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -p 8882:8882 marqoai/marqo:latest
```

## Papers by Author

Uses OpenAlex / pyalex to fetch from a gigantic free database of publications. One can fetch the top n papers by a authors name using `def get_papers_by_author(name: str) -> list[Query]:`.
