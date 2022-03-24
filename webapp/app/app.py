import time
import onnxruntime
from transformers import GPT2Tokenizer
from utils import generate
import streamlit as st

tokenizer = GPT2Tokenizer.from_pretrained('./tokenizer/', add_prefix_space=True)

session = onnxruntime.InferenceSession("./model/gpt3_custom.onnx")

model_params = {
    "num_attention_heads": 12,
    "hidden_size": 768,
    "num_layer": 12
}

st.title("Генератор русских народных сказок")

gen_len = st.sidebar.slider(label='Максимальная длинна текста', min_value=10, max_value=150, value=100, step=10)

body = st.text_input(label='Ключевые фразы', placeholder='Введите ключевые фразы, разделенные запятой', max_chars=200, key='keywords')

if body != '':    
    curr_time = time.time()
    promt = body.split(',')
    st.text(f'Затравка: {" _kw_ ".join(promt)}')
    text = generate(input_text=body, tokenizer=tokenizer, ort_session=session, num_tokens_to_produce=gen_len, **model_params)
    text = text.split('[SEP]')[-1].strip()
    st.text_area(label='Результат', value=text, disabled=True, height=200)
    st.text(f'Длительность: {str(round(time.time() - curr_time,2))} c')