import os

LOCAL_AI_LOCALHOST = 'http://localhost:8080'
LOCAL_AI_CODER = 'http://coder.aifb.kit.edu:8080'
LOCAL_AI_ML_PC = 'http://mlpc.coder.aifb.kit.edu:8080'

os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['OPENAI_BASE_URL'] = LOCAL_AI_LOCALHOST
os.environ['OPENAI_BASE_URL'] = LOCAL_AI_CODER
os.environ['OPENAI_BASE_URL'] = LOCAL_AI_ML_PC
# del os.environ['OPENAI_BASE_URL']
