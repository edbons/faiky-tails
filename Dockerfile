FROM python:3.8

WORKDIR /webapp

COPY webapp/requirements.txt ./

RUN pip install -r requirements.txt

COPY webapp/*.py ./
COPY webapp/model/ model/
COPY webapp/tokenizer/ tokenizer/

ENTRYPOINT [ "uvicorn" ]

CMD [ "--host", "0.0.0.0", "main:app" ]