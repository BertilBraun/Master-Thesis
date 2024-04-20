# Master's Thesis: Competence Extraction from Documents

## Overview

This project is part of my Master's thesis, which focuses on the extraction of competencies from academic documents using advanced AI techniques. The core objective is to develop and evaluate methods that automate the extraction of competencies from the abstracts and full texts of scientific papers. This involves summarizing documents, extracting competencies in various ways, and combining results using state-of-the-art language models.

## System Architecture

The system is built on a local AI server provided by the KIT AIFB. It utilizes multiple instances of language models to process and analyze texts through a retrieval-augmented setup. The architecture includes:

- **Chroma DB Database**: Integrates with a retrieval system to fetch relevant examples for competence extraction.
- **PyAlex Library**: Used to access the Open Alex dataset, which includes a wide range of academic publications.
- **PyPDF Library**: Powers the downloading and textual analysis of full-text documents.
- **Typed System**: Employs a strongly typed system for clean and manageable development.

## Components

The project includes several Python modules, each fulfilling specific roles within the system:

- **`database.py`**: Manages database interactions, particularly with Chroma DB for storing and retrieving document instances.
- **`evaluation.py`**: Handles the automatic evaluation of extracted competencies against model predictions.
- **`instance.py`**: Defines instances for different types of document processing and extraction tasks.
- **`language_model.py`**: Configures and manages interactions with the local AI language models.
- **`log.py`**: Provides logging functionality across various modules.
- **`openai_defines.py`**: Contains definitions and settings for OpenAI models used in the project.
- **`papers.py`**: Facilitates fetching and processing papers from the Open Alex dataset.
- **`types.py`**: Defines custom types and protocols for structured programming.
- **`util.py`**: Utility functions supporting various operations across modules.
- **`__main__.py`**: The entry point of the program, orchestrating the processing and evaluation workflows.

## Extraction Methods

The system explores three main extraction methods:

1. **Single-Prompt Extraction**: Extracts competencies from combined abstracts of multiple documents.
2. **Summarized Prompt Extraction**: Summarizes multiple papers into one prompt and then extracts competencies.
3. **Individual Paper Extraction**: Extracts competencies from each paper individually and combines the results.

## Evaluation

Competency profiles are generated for authors and evaluated in two main ways:

- **Automated Evaluation**: Uses a large language model to assess the match between extracted competencies and the actual content of the papers.
- **Expert Validation**: Profiles are sent to authors for ranking or scoring, which are then used to measure the accuracy of the extraction methods.

## Future Work

Plans include the use of reinforcement learning from automated feedback to fine-tune models based on their performance in competency extraction. This will also involve creating a synthetic dataset to improve and expand the training data available for models.

## Setup and Usage

To set up and run the system, ensure you have the necessary dependencies installed, then execute the main script:

```bash
python -m src
```

### Setup Local AI Server

Start the Local AI server using the following command:

```bash
docker run -p 8080:8080 --name local-ai -ti localai/localai:latest-aio-cpu
```

Access the Local AI server at `http://localhost:8080`.

Add a model to the Local AI server using the following command:

```bash
curl http://localhost:8080/models/apply -H "Content-Type: application/json" -d '{
     "url": "<MODEL_CONFIG_FILE>",
     "name": "<MODEL_NAME>"
    }'
```

Find the model configuration file [here](https://gitlab.kit.edu/kit/aifb/BIS/infrastruktur/localai/localai-model-gallery).

The used models in this project are:

```json
{
    "url": "https://gitlab.kit.edu/kit/aifb/BIS/infrastruktur/localai/localai-model-gallery/-/raw/main/text-embeddings.yaml",
    "name": "text-embedding-ada-002"
}
```

## Experimental Approaches and Alternatives

Throughout the development of this project, several experimental approaches were evaluated but ultimately not adopted for various reasons:

### LMQL (Language Model Query Language)

We explored using LMQL, a structured output formatting for large language models that enforces a user-defined structure. This method allows precise control over the output by masking potential outputs before sampling from the final distribution. However, this approach was found to be significantly slower—up to three times—compared to baseline methods. Given the structured output is nearly always followed when using carefully designed prompts, the additional complexity and processing time did not justify its use. Here is an example of how LMQL was intended to be used:

```bash
lmql serve-model (--dtype 8bit - only if a GPU is supported)
```

```python
import lmql

@lmql.query(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', is_async=False) # Example model from Hugging Face
def say(phrase):
    """lmql
    "Say '{phrase}': [TEST]" where len(TOKENS(TEST)) < 25
    return TEST
    """ # LMQL formatted prompt

print(say('Hello World!'))
```

LMQL was deemed not fast enough compared to traditional transformers with a 2-3x slowdown. The structured output is nearly always followed when using carefully designed prompts, so the additional complexity and processing time did not justify its use.

### Expected Output Format

The designed system aims to parse and extract data into a well-defined format, as illustrated below:

```markdown
Domain: Scientist in the Field of Machine Learning
Competencies:
- Computer Vision: Capability in developing machine vision systems for the recognition of objects, scenes, and activities in images and videos.
- Task Planning: Expertise in creating systems for efficient task planning and execution in dynamic environments.
- Autonomous Driving: Knowledge in the development of self-driving vehicles, including navigation, sensor technology, and decision-making processes.
- Analysis of Human Behavior: Experience in analyzing and interpreting human behavior using machine learning methods to enhance the interaction between humans and machines.
```

This format ensures clarity and consistency in the presentation of extracted competencies, which is essential for subsequent analyses and evaluations.

### Vector Database Alternatives: Marqo

Initially, the project also experimented with Marqo, a vector database, for embedding model flexibility and configuration. However, Marqo was eventually discarded in favor of Chroma DB due to the latter's better support for custom embedding models, which are crucial for our applications. Here are the basic commands for setting up Marqo:

```bash
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -p 8882:8882 marqoai/marqo:latest
```
