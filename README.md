# Domain-Agnostic Approaches to Competency Extraction via Large Language Models

## Overview

This repository hosts the implementation of the Master's Thesis titled "Domain-Agnostic Approaches to Competency Extraction via Large Language Models" by Bertil Braun, submitted to the Karlsruhe Institute of Technology (KIT). The thesis develops an innovative system utilizing Large Language Models (LLMs) to extract competencies from a variety of document types, improving upon existing methods that struggle with unstructured data across diverse domains.

Following the completion of the thesis, this work has been further developed into a scientific paper titled **"(Towards) Scalable Reliable Automated Evaluation with Large Language Models"** by Bertil Braun and Martin Forell, which has been accepted for publication.

## Thesis

The original thesis focuses on developing a competency extraction system that leverages LLMs to identify and extract competency profiles from diverse document types. The system addresses key challenges in processing unstructured data across different domains and provides a robust framework for competency analysis.

**Key contributions of the thesis:**
- Multi-phase approach for competency extraction using LLMs
- Advanced fine-tuning methodologies including Direct Preference Optimization (DPO)
- Comprehensive evaluation framework combining expert and automatic assessment
- Domain-agnostic system capable of handling various document types

The complete thesis document is available at `documentation/Master_Thesis.pdf`.

## Published Paper: "(Towards) Scalable Reliable Automated Evaluation with Large Language Models"

Building upon the thesis work, the research has evolved into a scientific paper that introduces a novel evaluation framework for assessing LLM-generated content. This paper addresses one of the core challenges identified during the thesis research: the need for reliable, scalable evaluation methods.

### Paper Abstract

Evaluating the quality and relevance of textual outputs from Large Language Models (LLMs) remains challenging and resource-intensive. Existing automated metrics often fail to capture the complexity and variability inherent in LLM-generated outputs. Moreover, these metrics typically rely on explicit reference standards, limiting their use mostly to domains with objective benchmarks. This work introduces a novel evaluation framework designed to approximate expert-level assessments of LLM-generated content. The proposed method employs pairwise comparisons of outputs by multiple LLMs, reducing biases from individual models. An Elo rating system is used to generate stable and interpretable rankings. Adjustable agreement thresholds—from full unanimity to majority voting—allow flexible control over evaluation confidence and coverage. The method's effectiveness is demonstrated through evaluating competency profiles extracted from scientific abstracts. Preliminary results show that automatically derived rankings correlate well with expert judgments, significantly reducing the need for extensive human intervention. By offering a scalable, consistent, and domain-agnostic evaluation layer, the framework supports more efficient and reliable quality assessments of LLM outputs across diverse applications.

### Key Innovations of the Paper

- **Pairwise Comparison Framework**: Employs multiple LLMs to perform pairwise comparisons, reducing individual model biases
- **Elo Rating System**: Generates stable and interpretable rankings for LLM outputs
- **Flexible Agreement Thresholds**: Allows control over evaluation confidence through adjustable consensus requirements
- **Domain-Agnostic Evaluation**: Provides scalable assessment without requiring explicit reference standards
- **Expert-Level Approximation**: Demonstrates correlation with human expert judgments while significantly reducing manual intervention

The complete paper is available at `documentation/Towards Scalable Reliable Automated Evaluation with Large Language Models.pdf` and the conference poster at `documentation/Poster - Towards Scalable Reliable Automated Evaluation with Large Language Models.pdf`.

### Conference Poster

![Conference Poster](documentation/Poster%20-%20Towards%20Scalable%20Reliable%20Automated%20Evaluation%20with%20Large%20Language%20Models.jpg)

## Releases

Two major releases are available:

1. **Thesis Release**: Contains the final version of the Master's Thesis
2. **Paper Release**: Includes the final thesis, published paper, and conference poster

## System Description

The competency extraction system is built on a multi-phase approach that includes selecting, fine-tuning, and evaluating LLMs across different types of documents to create accurate competency profiles. This section provides an overview of each major component of the system:

![System Overview](documentation/Data%20Flow%20Extraction.png)

### Data Processing and Summarization

The process begins with the collection of input documents that are mostly based on papers by authors from various domains. These documents are fetched and preprocessed in `src/logic/papers.py`, which includes the extraction of relevant text and metadata.

### Competency Extraction

Extracted summaries are then processed to identify and extract competency profiles. This extraction is performed using LLMs designed to pull relevant competencies from the documents. Each document's competencies are initially profiled individually in `src/extraction` using three different extraction methods.

### Model Fine-Tuning

The system employs advanced fine-tuning methodologies, Direct Preference Optimization (DPO), to adapt the LLMs to the specific task of competency extraction. The fine-tuning process, located in `src/finetuning`, optimizes the models to enhance their performance across various document types and domains by learning from synthetic data. The fine-tuning process is designed to be run on the BW-UniCluster, a high-performance computing cluster. SLURM scripts for running the setup and fine-tuning process are located in `src/finetuning`. For more information on the fine-tuning process, refer to `src/finetuning/README.md`.

### Evaluation Framework

To validate the accuracy of the extracted profiles, the system incorporates both expert and automatic evaluation mechanisms. These evaluations compare the competencies extracted by the system against benchmarks set by human experts and automated systems to ensure reliability and accuracy. The evaluation scripts are found in `src/scripts/automatic_evaluation_correlation_analysis.py`.

The evaluation methodology developed for this thesis has been significantly enhanced and forms the foundation of the published paper's novel evaluation framework.

### Visualization and Reporting

For easy interpretation and analysis, the system generates visualizations and structured reports of the competency profiles. This functionality, designed to help users quickly understand and utilize the extracted data, is handled by templates in `src/templates` and generated by `src/logic/display.py`.

## Installation and Setup

To setup the system, follow these steps:

```bash
# Clone the repository
git clone https://github.com/BertilBraun/Master-Thesis.git Master-Thesis
# Navigate to the project directory
cd Master-Thesis
# Install the required dependencies
pip install -r requirements.txt
```

The system requires Python 3.11 or higher to run as a result of the strong type annotations added.

Furthermore, to be able to use the system, you need to add API keys for OpenAI and JsonBin.io to the `src/defines.py` file.

For the fine-tuning process, you need to have access to the BW-UniCluster. Other high-performance computing clusters can be used as well, but the SLURM scripts might need to be adjusted accordingly. For a detailed guide on how to set up the fine-tuning process, refer to `src/finetuning/README.md`.

## Documentation

Detailed documentation of the system and its components is available within the repository. This includes:

- **Master's Thesis**: `documentation/Master_Thesis.pdf` - Complete thesis document
- **Published Paper**: `documentation/Towards Scalable Reliable Automated Evaluation with Large Language Models.pdf` - Scientific paper on scalable automated evaluation
- **Conference Poster**: `documentation/Poster - Towards Scalable Reliable Automated Evaluation with Large Language Models.pdf` - Visual summary of the paper's key contributions
- Additional supporting materials and references

## License

This project is licensed under the MIT License, which allows for extensive reuse and modification in academic and commercial projects.
