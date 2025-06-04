import pandas as pd
import numpy as np
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import pickle
import os
from utils import load_data
from preprocess import preprocess_data

# Загрузка данных
data_path = "data/raw/student_data.csv"
df = load_data(data_path)

# Препроцессинг
df = preprocess_data(df)

# Определение признаков и целевой переменной
features = [
    "online_activity",
    "assignments_completed",
    "grades",
    "forum_participation",
    "activity_trend",
    "assignments_trend",
    "grades_trend",
    "activity_score",
    "activity_ratio",
    "age",
    "region",
    "semester",
]
target = "churn"

# Разделение на train/test по student_id
student_ids = df["student_id"].unique()
train_ids, test_ids = train_test_split(student_ids, test_size=0.2, random_state=42)
train_data = df[df["student_id"].isin(train_ids)]
test_data = df[df["student_id"].isin(test_ids)]

# Настройка задачи
task = Task("binary", metric="auc")

# Настройка LightAutoML
automl = TabularAutoML(
    task=task,
    timeout=1800,
    cpu_limit=4,
    general_params={"use_algos": [["catboost", "lgb", "xgboost"]]},
    reader_params={
        "cv": 5,
        "random_state": 42,
        "advanced_roles_params": {"class_balancing": True},
    },
    tuning_params={"max_tuning_iter": 50},
)
# Обучение модели
oof_pred = automl.fit_predict(
    train_data, roles={"target": target, "category": ["region"]}, verbose=3
)

# Оценка на тестовой выборке
y_true = test_data[target]
y_prob = automl.predict(test_data[features]).data.ravel()

# Проверка метрик для разных порогов
thresholds = [0.2, 0.3, 0.4]
for thresh in thresholds:
    y_pred = (y_prob > thresh).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"Threshold {thresh}: Precision = {precision:.2f}, Recall = {recall:.2f}")

# Сохранение модели
os.makedirs("models", exist_ok=True)
with open("models/churn_model.pkl", "wb") as f:
    pickle.dump(automl, f)
print("Модель сохранена в models/churn_model.pkl")
