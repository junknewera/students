# Churn Prediction Project
Проект для предсказания оттока студентов с использованием ML.
## Структура
- `data/raw/`: Исходные данные
- `data/processed/`: Обработанные данные
- `src/`: Скрипты для препроцессинга, обучения и предсказания
- `docker/`: Файлы для контейнеризации
- `airflow/`: DAG для Airflow
## Установка
1. Установить зависимости: `pip install -r requirements.txt`
2. Запустить препроцессинг: `python src/data/preprocess.py`