import os
import dotenv

dotenv.load_dotenv()


def __get(name: str, default: str | None = None) -> str:
    value = os.getenv(name)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f'Please set the {name} environment variable')
    return value


OPENAI_API_KEY = __get('OPENAI_API_KEY')
JSONBIN_API_KEY = __get('JSONBIN_API_KEY')
GROQ_API_KEY = __get('GROQ_API_KEY')
GOOGLE_API_KEY = __get('GOOGLE_API_KEY')

GROQ_BASE_URL = 'https://api.groq.com/openai/v1'
GOOGLE_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

DEBUG = False  # Set to True to enable debugging output (streaming of the AI's output to the console)
MAX_RETRIES = 1

LOCAL_AI_LOCALHOST = 'http://localhost:8080'
LOCAL_AI_CODER = 'http://coder.aifb.kit.edu:8080'
LOCAL_AI_ML_PC = 'http://mlpc.coder.aifb.kit.edu:8080'
LOCAL_AI_ML_PC = 'http://aifb-bis-gpu01.aifb.kit.edu:8080'

# BASE_URL_LLM = None # When using the OpenAI API instead of the Local AI
BASE_URL_LLM = LOCAL_AI_ML_PC

BASE_URL_LLM = GROQ_BASE_URL
OPENAI_API_KEY = GROQ_API_KEY
