from fastapi import FastAPI, Response
from pydantic import BaseModel

from transformers import GPT2Tokenizer
import onnxruntime
from utils import test_generation

tokenizer = GPT2Tokenizer.from_pretrained('./tokenizer/', add_prefix_space=True)

session = onnxruntime.InferenceSession("./model/gpt3_custom.onnx")

model_params = {
    "num_attention_heads": 12,
    "hidden_size": 768,
    "num_layer": 12
}

class Body(BaseModel):
    phrase: str


app = FastAPI()


@app.get('/')
def root():
    return Response("<h1>A self-documenting API to interact with an ONNX model</h1>")


@app.post('/generate', description='Return generated text. For input, post keywords delimited with commas')
def generate(body: Body):
    promt = body.phrase.split(',')
    output = test_generation(tokenizer, promt, num_tokens_to_produce=100, ort_session=session, **model_params)
    result = tokenizer.decode(output['input_ids'][0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    return {'text': result}