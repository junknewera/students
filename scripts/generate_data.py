import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Параметры
n_students = 34000  # Уникальных студентов
semester_probs = [0.2, 0.15, 0.15, 0.5]  # Вероятности для 1, 2, 3, 4 семестров
total_records = 100000  # Цель по записям


# Функция для определения количества семестров
def get_semesters_completed():
    return np.random.choice([1, 2, 3, 4], p=semester_probs)


# Генерация данных
data = []
record_count = 0

for student_id in range(1, n_students + 1):
    if record_count >= total_records:
        break
    semesters = get_semesters_completed()
    age = np.random.randint(22, 51)
    region = np.random.choice(["Москва", "Сибирь", "Урал", "Юг"])
    base_activity = np.clip(np.random.normal(70, 15), 0, 100)
    base_grades = np.clip(np.random.normal(3.5, 0.7), 0, 5)
    base_forum = np.clip(np.random.poisson(5), 0, 10)

    for semester in range(1, semesters + 1):
        if record_count >= total_records:
            break
        # Вариации для каждого семестра
        online_activity = np.clip(np.random.normal(base_activity, 5), 0, 100)
        assignments_completed = np.clip(np.random.normal(online_activity, 10), 0, 100)
        grades = np.clip(np.random.normal(base_grades, 0.2), 0, 5)
        forum_participation = np.clip(np.random.normal(base_forum, 1), 0, 10)

        # Уменьшение активности перед оттоком
        if semester == semesters and semesters < 4:
            online_activity *= 0.8
            assignments_completed *= 0.8
            grades *= 0.9
            churn = 1
        else:
            churn = 0

        data.append(
            {
                "student_id": f"S{student_id:05d}",
                "semester": semester,
                "age": age,
                "region": region,
                "online_activity": round(online_activity, 2),
                "assignments_completed": round(assignments_completed, 2),
                "grades": round(grades, 2),
                "forum_participation": round(forum_participation, 2),
                "churn": churn,
            }
        )
        record_count += 1

# Создание DataFrame
df = pd.DataFrame(data)

# Сохранение в data/raw/
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/student_data.csv", index=False)

print(f"Сгенерировано {len(df)} записей")
print(f"Доля оттока: {df['churn'].mean():.2%}")
print(f"Уникальных студентов: {df['student_id'].nunique()}")
