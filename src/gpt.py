import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import openai

from transformers import TextGenerationPipeline, pipeline

from src.log import log, LogLevel
from src.util import timeit


generators: dict[str, TextGenerationPipeline] = {}


@timeit('Querying transformers')
def query_transformers(
    prompt: str,
    model: str,
    max_new_tokens: int = 250,
    num_return_sequences: int = 1,
    stop_sequence: str = '\n\n',
) -> str:
    if model not in generators:
        generators[model] = pipeline('text-generation', model=model)  # type: ignore

    # Generate the response
    response = generators[model](
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        # stop_sequence=stop_sequence, # Seems to cut off the response early
    )

    generated_text: str = response[0]['generated_text']  # type: ignore

    log('Query Transformer generated Text:', generated_text, level=LogLevel.DEBUG)

    generated_text = generated_text.replace(prompt, '')  # Remove the prompt from the generated text

    return generated_text


@timeit('Querying OpenAI')
def query_openai(
    prompt: str,
    model: str = 'GPT4',
    max_new_tokens: int = 250,
    num_return_sequences: int = 1,
    stop_sequence: str = '\n\n',
) -> str:
    generated_text = (
        openai.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
            stop=stop_sequence,
        )
        .choices[0]
        .text
    )

    log('Query OpenAI generated Text:', generated_text, level=LogLevel.DEBUG)

    return generated_text
