# Faiky-tails

В проекте реализована генерация русских народных сказок. Полный отчет приведен в каталоге docs/.

Дообучено две модели *[SberAI GPT3 Small](https://github.com/sberbank-ai/ru-gpts)*:

* Модель №1. Из текстов автоматически извлекаются N-граммы ключевых фраз алгоритмом *[RAKE](https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents)*
* Модель №2. Дополнительно к варианту №1 из текстов автоматически извлекаются именованные сущности (имена главных героев) средствами библиотеки spacy.

Веса, результаты обучения моделей и сгенерированные истории можно скачать с *[гугл диск](https://drive.google.com/file/d/1f1MU0bgIo1X_78vpuc-DqKH8joHRcbgT/view?usp=sharing)*.

## Запуск в Docker

1. Запустить контейнеры и сборку локальных образов Docker:

```
docker compose -f docker-compose.yaml up -d --build
```

2. Подключиться к REST API сервису по адресу: http://localhost:8000/docs
3. Подключиться к приложению streamlit по адресу: http://localhost:8501

## Запуск в k8s

1. Запустить сборку локальных образов Docker:
```
docker build -f webapp/Dockerfile_app --tag faikytail/app --no-cache webapp
docker build -f webapp/Dockerfile_api --tag faikytail/api --no-cache webapp
```

2. Запустить контейнеры k8s:

```
kubectl apply -f webapp.yaml
```

3. Подключиться к приложению streamlit по адресу: http://localhost:8080

## TO DO

* попробовать метрики, измеряющие насколько префикс генерации учтен в сгенерированном тексте.
* попробовать другие подходы по управляемой генерации историй.
