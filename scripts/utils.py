import pandas as pd
import os


def load_data(file_path):
    """Загрузка данных из CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    return pd.read_csv(file_path)


def save_data(df, file_path):
    """Сохранение данных в CSV."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
