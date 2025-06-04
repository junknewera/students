import numpy as np
import pandas as pd
import os
from pathlib import Path

# Настройка генератора случайных чисел
np.random.seed(42)

# Параметры
n_students = 10000
semesters_per_student = np.random.randint(1, 9, size=n_students)  # 1–8 семестров
churn_rate = 0.225  # 22.5% оттока
regions = ["Moscow", "SPb", "Kazan", "Novosibirsk", "Ekaterinburg"]
faculties = ["IT", "Economics", "Engineering", "Humanities", "Medicine"]
campuses = ["Central", "North", "South"]

# Инициализация списков
data = {
    "student_id": [],
    "semester": [],
    "age": [],
    "region": [],
    "attendance": [],
    "avg_grade": [],
    "lms_logins": [],
    "lms_tasks_done": [],
    "faculty": [],
    "campus": [],
    "tuition_fee": [],
    "initial_discount": [],
    "lms_activity_score": [],
    "grade_trend": [],
    "attendance_drop": [],
    "churn": [],
}

# Генерация данных
for student_id in range(10000, 10000 + n_students):
    n_semesters = semesters_per_student[student_id - 10000]
    is_churn = np.random.binomial(1, churn_rate) if n_semesters > 1 else 0

    for semester in range(1, n_semesters + 1):
        data["student_id"].append(student_id)
        data["semester"].append(semester)
        data["age"].append(int(np.random.normal(21, 1.5)))
        data["region"].append(np.random.choice(regions))
        # Посещаемость с пропусками
        attendance = np.random.normal(85, 10) if np.random.random() > 0.05 else np.nan
        data["attendance"].append(max(0, min(100, attendance)))
        # Средний балл с пропусками
        avg_grade = np.random.normal(75, 10) if np.random.random() > 0.03 else np.nan
        data["avg_grade"].append(max(0, min(100, avg_grade)))
        # LMS-активность
        lms_logins = np.random.poisson(50)
        lms_tasks_done = np.random.poisson(30)
        data["lms_logins"].append(lms_logins)
        data["lms_tasks_done"].append(lms_tasks_done)
        data["faculty"].append(np.random.choice(faculties))
        data["campus"].append(np.random.choice(campuses))
        data["tuition_fee"].append(
            max(50000, min(200000, np.random.normal(120000, 30000)))
        )
        data["initial_discount"].append(max(0, min(30, np.random.normal(10, 5))))
        # LMS activity score (нормализованный)
        lms_activity = (lms_logins / 100 + lms_tasks_done / 50) / 2
        data["lms_activity_score"].append(lms_activity)
        # Grade trend и attendance drop (0 для первого семестра)
        if semester == 1:
            data["grade_trend"].append(0)
            data["attendance_drop"].append(0)
        else:
            prev_grade = (
                data["avg_grade"][-2] if not np.isnan(data["avg_grade"][-2]) else 75
            )
            prev_attendance = (
                data["attendance"][-2] if not np.isnan(data["attendance"][-2]) else 85
            )
            data["grade_trend"].append(
                avg_grade - prev_grade if not np.isnan(avg_grade) else 0
            )
            data["attendance_drop"].append(
                prev_attendance - attendance if not np.isnan(attendance) else 0
            )
        # Churn (только в последнем семестре)
        data["churn"].append(1 if semester == n_semesters and is_churn else 0)

# Создание DataFrame
df = pd.DataFrame(data)

# Создание директории и сохранение
Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/synthetic_students.csv", index=False)

print(
    f"Сгенерировано {len(df)} записей для {n_students} студентов. Файл сохранён в data/raw/synthetic_students.csv"
)
