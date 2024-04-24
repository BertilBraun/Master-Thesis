import os
import dotenv

dotenv.load_dotenv()


DEBUG = False

LOCAL_AI_LOCALHOST = 'http://localhost:8080'
LOCAL_AI_CODER = 'http://coder.aifb.kit.edu:8080'
LOCAL_AI_ML_PC = 'http://mlpc.coder.aifb.kit.edu:8080'

LOCAL_AI_ML_PC = LOCAL_AI_CODER  # TODO temporary until coder is back up

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-...')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# BASE_URL_* = None # When using the OpenAI API instead of the Local AI
BASE_URL_EMBEDDINGS = LOCAL_AI_CODER
BASE_URL_LLM = LOCAL_AI_ML_PC
