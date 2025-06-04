import pandas as pd
import numpy as np


def preprocess_data(df):
    """Препроцессинг данных: фича-инжиниринг и обработка пропусков."""
    df = df.copy()

    # Сортировка по student_id и semester для корректного расчёта трендов
    df = df.sort_values(["student_id", "semester"])

    # Фича-инжиниринг
    df["activity_trend"] = df.groupby("student_id")["online_activity"].diff().fillna(0)
    df["assignments_trend"] = (
        df.groupby("student_id")["assignments_completed"].diff().fillna(0)
    )
    df["grades_trend"] = df.groupby("student_id")["grades"].diff().fillna(0)
    df["activity_score"] = (
        df["online_activity"] * 0.4
        + df["assignments_completed"] * 0.4
        + df["forum_participation"] * 2
    )
    df["activity_ratio"] = df["assignments_completed"] / (
        df["online_activity"] + 1e-6
    )  # Избегаем деления на 0

    # Обработка пропусков
    numeric_cols = [
        "online_activity",
        "assignments_completed",
        "grades",
        "forum_participation",
        "activity_ratio",
    ]
    for col in numeric_cols:
        df[col] = df[col].fillna(df.groupby("semester")[col].transform("median"))

    return df
