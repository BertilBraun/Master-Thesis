import os
import dotenv

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-...')
JSONBIN_API_KEY = os.getenv('JSONBIN_API_KEY', '$2a$...')
JSONBIN_API_KEY = '$2a$10$e5lgWuePR6yPBPIBgN8fb.iIRqRnYXQ9mv2iUL3466zUz0p9CMeVe'

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


DEBUG = False  # Set to True to enable debugging output (streaming of the AI's output to the console)
MAX_RETRIES = 1

LOCAL_AI_LOCALHOST = 'http://localhost:8080'
LOCAL_AI_CODER = 'http://coder.aifb.kit.edu:8080'
LOCAL_AI_ML_PC = 'http://mlpc.coder.aifb.kit.edu:8080'
LOCAL_AI_ML_PC = 'http://aifb-bis-gpu01.aifb.kit.edu:8080'


# BASE_URL_LLM = None # When using the OpenAI API instead of the Local AI
BASE_URL_LLM = LOCAL_AI_ML_PC
