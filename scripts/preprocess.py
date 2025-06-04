# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Основная функция предобработки данных"""
    df = df.copy()

    # 1. Сортировка по student_id и semester
    df = df.sort_values(["student_id", "semester"])

    # 2. Фича-инжиниринг
    df = _generate_features(df)

    # 3. Обработка пропусков
    df = _handle_missing_values(df)

    # 4. Кодирование категориальных признаков
    df, ohe = _encode_categorical(df)

    # 5. Разделение на train/test по student_id
    train_df, test_df = _train_test_split(df, test_size)

    return train_df, test_df, ohe


def _generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генерация новых признаков"""
    # Тренды активности
    df["activity_trend"] = df.groupby("student_id")["online_activity"].diff().fillna(0)
    df["assignments_trend"] = (
        df.groupby("student_id")["assignments_completed"].diff().fillna(0)
    )

    # Комбинированные метрики
    df["activity_score"] = (
        0.4 * df["online_activity"]
        + 0.4 * df["assignments_completed"]
        + 0.2 * df["forum_participation"]
    )

    # Кумулятивные показатели
    df["cumulative_grades"] = df.groupby("student_id")["grades"].transform(
        lambda x: x.expanding().mean()
    )

    # Флаги аномалий
    df["low_activity_flag"] = (
        (df["online_activity"] < 30) | (df["assignments_completed"] < 40)
    ).astype(int)

    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Заполнение пропусков"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = df.groupby("semester")[col].transform(lambda x: x.fillna(x.median()))
    return df


def _encode_categorical(df: pd.DataFrame) -> tuple:
    """One-Hot Encoding для региона и факультета"""
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_cols = ["region", "faculty"]
    ohe_features = ohe.fit_transform(df[cat_cols])

    # Создаем DF с новыми признаками
    ohe_df = pd.DataFrame(
        ohe_features, columns=ohe.get_feature_names_out(cat_cols), index=df.index
    )

    # Удаляем исходные колонки и объединяем с OHE
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, ohe_df], axis=1)

    return df, ohe


def _train_test_split(df: pd.DataFrame, test_size: float) -> tuple:
    """Разделение по student_id (без утечки данных)"""
    student_ids = df["student_id"].unique()
    train_ids, test_ids = train_test_split(
        student_ids, test_size=test_size, random_state=42
    )

    train_df = df[df["student_id"].isin(train_ids)]
    test_df = df[df["student_id"].isin(test_ids)]

    return train_df, test_df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/student_data.csv")
    train_df, test_df, ohe = preprocess_data(df)
    print(train_df.shape, test_df.shape)
