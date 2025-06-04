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
from scipy.special import expit  # Сигмоидная функция

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
    "log_activity_score",
    "activity_forum_interaction",
    "cumulative_grades",
    "low_activity_flag",
    "age",
    "semester",
    "region",
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
    timeout=1800,  # 30 минут
    cpu_limit=4,
    general_params={"use_algos": [["catboost", "lgb", "xgboost"]], "nn_models": None},
    reader_params={
        "cv": 5,
        "random_state": 42,
        "class_weights": {0: 1, 1: 5},
    },  # Балансировка классов
    tuning_params={"max_tuning_iter": 10},
)

# Обучение модели
train_data_subset = train_data[features + [target]]
oof_pred = automl.fit_predict(
    train_data_subset, roles={"target": target, "category": ["region"]}, verbose=1
)


# Калибровка вероятностей (сигмоидная функция)
def calibrate_probs(raw_probs):
    return expit(raw_probs)


# Оценка на тестовой выборке
y_true = test_data[target]
raw_probs = automl.predict(test_data[features]).data
y_prob = calibrate_probs(raw_probs)

# Проверка метрик для разных порогов
thresholds = [0.1, 0.2, 0.3, 0.4]
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
