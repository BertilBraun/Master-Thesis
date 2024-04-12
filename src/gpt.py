import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


class KeywordsStoppingCriteria(StoppingCriteria):
    # Stop generation when the last token is in the keywords list

    def __init__(self, keywords_ids: list[int]):
        self.keywords = keywords_ids

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


class Model:
    def __init__(self, model: str):
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, pad_token_id=self.tokenizer.eos_token_id).eval()

    def generate(self, prompt: str, max_new_tokens: int, num_return_sequences: int, stop_sequence: str) -> str:
        keywords = self.tokenizer.encode(stop_sequence)
        stopping_criteria = KeywordsStoppingCriteria(keywords)

        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

        output = self.model.generate(
            inputs,
            max_length=len(inputs['input_ids'][0]) + max_new_tokens,  # type: ignore
            num_return_sequences=num_return_sequences,
            stopping_criteria=StoppingCriteriaList([stopping_criteria]),
        )

        return self.tokenizer.decode(*output)
