import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pathlib import Path
import os
from dotenv import load_dotenv

# Загрузка конфигурации
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data/raw/synthetic_students.csv")
PROCESSED_PATH = os.getenv("PROCESSED_PATH", "data/processed/processed_students.csv")

# Чтение данных
df = pd.read_csv(DATA_PATH)

# Заполнение пропусков
# Attendance: медиана по факультету
df["attendance"] = df.groupby("faculty")["attendance"].transform(
    lambda x: x.fillna(x.median())
)
# Avg_grade: среднее по студенту
df["avg_grade"] = df.groupby("student_id")["avg_grade"].transform(
    lambda x: x.fillna(x.mean())
)
# Если пропуски остались (нет данных по студенту), заполняем глобальной медианой
df["attendance"].fillna(df["attendance"].median(), inplace=True)
df["avg_grade"].fillna(df["avg_grade"].mean(), inplace=True)

# Проверка и пересчёт признаков (для демонстрации)
df["lms_activity_score"] = (
    df["lms_logins"] / df["lms_logins"].max()
    + df["lms_tasks_done"] / df["lms_tasks_done"].max()
) / 2
df["grade_trend"] = df.groupby("student_id")["avg_grade"].diff().fillna(0)
df["attendance_drop"] = -df.groupby("student_id")["attendance"].diff().fillna(0)

# Кодирование категориальных признаков
categorical_cols = ["region", "faculty", "campus"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_cats = pd.DataFrame(
    encoder.fit_transform(df[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
)
df = df.drop(categorical_cols, axis=1).join(encoded_cats)

# Нормализация числовых признаков
numeric_cols = [
    "age",
    "attendance",
    "avg_grade",
    "lms_logins",
    "lms_tasks_done",
    "tuition_fee",
    "initial_discount",
    "lms_activity_score",
    "grade_trend",
    "attendance_drop",
]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Сохранение обработанных данных
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

print(f"Обработанные данные сохранены в {PROCESSED_PATH}")
