from src.hyperparameter_defines import REFERENCE_GENERATION_MODEL, EVALUATION_MODEL, OTHER_REFERENCE_GENERATION_MODEL


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 2:
        sys.exit('Usage: python -m src <command> <query>')

    if sys.argv[1] == 'gen_example':
        from src.extraction.references.generate_references import generate_example_references

        generate_example_references(int(sys.argv[2]), REFERENCE_GENERATION_MODEL)

    if sys.argv[1] == 'gen_combination':
        from src.extraction.references.generate_references import generate_combination_references

        generate_combination_references(int(sys.argv[2]), REFERENCE_GENERATION_MODEL)

    if sys.argv[1] == 'gen_ranking':
        from src.extraction.references.generate_references import generate_ranking_references

        generate_ranking_references(int(sys.argv[2]), EVALUATION_MODEL, OTHER_REFERENCE_GENERATION_MODEL)
