# Paper Evaluation Extension Server

This is a Flask application intended for additional expert evaluation. It provides endpoints for paper evaluation and processing.

## How to Run

0. **Install Dependencies**: Install the dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

   Make sure to have `cuda` installed on your system and are using a GPU.

1. **Set Up API Keys**: Define your GROQ, OpenAI and JsonBin API keys in `src/defines.py`.

2. **Start the Server**: From the root directory of the project, run:

   ```bash
   python -m src.paper_evaluation_extension.server
   ```

3. **Access the Server**: The server will be running at `http://localhost:5000/index`.
