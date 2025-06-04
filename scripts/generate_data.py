import pandas as pd
import numpy as np
from faker import Faker

# Инициализация генератора
fake = Faker("ru_RU")
np.random.seed(42)

# Параметры генерации
n_students = 34000
max_semesters = 4

# Генерация базовых данных
students = []
for student_id in range(1, n_students + 1):
    start_year = np.random.randint(2020, 2023)
    region = fake.random_element(elements=("Москва", "СПб", "Сибирь", "Урал", "Юг"))
    faculty = fake.random_element(
        elements=("Информатика", "Экономика", "Юриспруденция", "Лингвистика")
    )

    for semester in range(1, max_semesters + 1):
        # Динамическое ухудшение показателей перед оттоком
        if semester == max_semesters and np.random.rand() < 0.25:
            churn = 1
            online_activity = max(10, np.random.normal(30, 10))
            assignments = max(15, np.random.normal(40, 15))
        else:
            churn = 0
            online_activity = np.random.normal(70, 15)
            assignments = np.random.normal(80, 10)

        students.append(
            {
                "student_id": student_id,
                "semester": semester,
                "age": np.random.randint(18, 35),
                "region": region,
                "faculty": faculty,
                "online_activity": max(0, min(100, online_activity)),
                "assignments_completed": max(0, min(100, assignments)),
                "grades": max(2.0, min(5.0, np.random.normal(4.0, 0.7))),
                "forum_participation": np.random.poisson(3),
                "churn": churn,
            }
        )

# Создание DataFrame
df = pd.DataFrame(students)
df.to_csv("data/raw/student_data.csv", index=False)
