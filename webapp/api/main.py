from fastapi import FastAPI, Response
from pydantic import BaseModel
from transformers import GPT2Tokenizer
import onnxruntime
from utils import generate

tokenizer = GPT2Tokenizer.from_pretrained('./tokenizer/', add_prefix_space=True)

session = onnxruntime.InferenceSession("./model/gpt3_custom.onnx")

model_params = {
    "num_attention_heads": 12,
    "hidden_size": 768,
    "num_layer": 12
}

class Body(BaseModel):
    phrase: str
    max_len: int
    class Config:
        schema_extra = {
            "example": {
                "phrase": "Жил был старик",
                "max_len": 50                
            }
        }

app = FastAPI()

@app.get('/')
def root():
    return Response("<h1>A self-documenting API to interact with an ONNX model</h1>")

@app.post('/story', description='Return generated text of story. For input, post keywords delimited with commas')
def generate_story(body: Body):
    promt = body.phrase.split(',')
    text = generate(input_text=promt, tokenizer=tokenizer, ort_session=session, num_tokens_to_produce=body.max_len, **model_params)
    text = text[0].split('[SEP]')[-1].strip()  # return only first hypothesis
    return {'text': text}