import time
import streamlit as st
import requests

API_URL = 'http://faikytail-api:8000/story'

st.title("Генератор русских народных сказок")

gen_len = st.sidebar.slider(label='Максимальная длинна текста', min_value=10, max_value=150, value=100, step=10)

body = st.text_input(label='Ключевые фразы', placeholder='Введите ключевые фразы, разделенные запятой', max_chars=200, key='keywords')

if body != '':    
    curr_time = time.time()
    
    json = {
            "phrase": body,
            "max_len": gen_len
            }
    
    response = requests.post(url=API_URL, json=json)

    if response.status_code in [200, 201]:
        text = response.json()['text']
    else:
        text = f'Sorry, something goes wrong... Request response {response.status_code} {response.reason}'
    
    st.text_area(label='Результат', value=text, disabled=True, height=200)
    st.text(f'Длительность: {str(round(time.time() - curr_time,2))} c')