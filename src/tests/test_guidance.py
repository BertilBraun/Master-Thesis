from guidance import models, gen
import time

MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'  # 'meta-llama/Llama-2-7b-chat-hf'  # Example model specification

model = models.Transformers(MODEL)

start_time = time.time()

response = (
    model + "I'm going on a beach vacation and need to pack. List the items I should bring: " + gen(max_tokens=100)
)


print('Inference time:', time.time() - start_time)
print(response)
