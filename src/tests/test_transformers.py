from transformers import pipeline, set_seed
import time


def generate_packing_list():
    # Initialize the generator
    generator = pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    set_seed(42)

    abstract = 'Investigating the efficacy of machine learning algorithms in predicting stock market trends, this paper presents a model that outperforms traditional analytical methods.'

    # Define the prompt
    prompt = f"""Examples:

---

Example 1:

Abstract: "Through the application of deep learning techniques to satellite imagery, this research uncovers new patterns in urban development, contributing to more sustainable city planning."

Domain: "Expert in applying AI for sustainable urban development."

Competencies:
- AI in Urban Planning: Utilizes deep learning to analyze satellite images for city planning.
- Sustainable Development: Innovates in sustainable urban development strategies.
- Pattern Recognition: Identifies key urban development patterns using AI.
- Data Analysis: Expert in analyzing large-scale geographical data.

Example 2:

Abstract: "Examining social media's impact on political discourse, this study employs natural language processing (NLP) to analyze sentiment and influence in online discussions, shedding light on digital communication's role in shaping public opinion."

Domain: "Specialist in digital communication and political discourse analysis."

Competencies:
- NLP and Sentiment Analysis: Applies NLP to understand social media influence.
- Digital Communication: Studies the impact of online platforms on communication.
- Public Opinion Research: Analyzes how digital discourse shapes political opinions.
- Data-Driven Insights: Generates insights into political discussions using data analysis.

---

Task Description:

Extract and summarize key competencies from scientific paper abstracts, aiming for a general overview suitable across disciplines. Begin with a concise Domain that captures the main area of expertise in about ten words, abstract enough to apply broadly within a scientific context. Then, list three to eight specific competencies with brief descriptions based on the abstract.

The following is now your task. Please generate a Domain and competencies based on the following abstract. Do not generate anything except the Domain and competencies based on the abstract.

Abstract: "{abstract}"

"""

    # Start timing
    start_time = time.time()

    # Generate the response
    response = generator(prompt, max_length=800, num_return_sequences=1)

    # End timing
    end_time = time.time()

    # Calculate inference time
    inference_time = end_time - start_time

    # Print the generated packing list
    print(response[0]['generated_text'])  # type: ignore

    # Print the inference time
    print(f'Inference time: {inference_time} seconds')


if __name__ == '__main__':
    generate_packing_list()
    generate_packing_list()
    generate_packing_list()
    generate_packing_list()
    generate_packing_list()
