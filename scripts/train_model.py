import pandas as pd
import numpy as np
import torch
import pickle
import os
from imblearn.over_sampling import SMOTE
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.metrics import precision_score, recall_score
from preprocess import preprocess_data


def train_model():
    # Constants
    RANDOM_STATE = 42
    N_THREADS = 4
    N_FOLDS = 5
    TARGET_NAME = "churn"
    TIMEOUT = 300
    MODEL_PATH = "models/automl_model.pkl"

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # ... (rest of your existing code remains the same until the metrics printing)

    print("\nModel Performance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Save the model
    print(f"\nSaving model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(automl, f)
    print("Model saved successfully!")


def train_model():
    # Constants
    RANDOM_STATE = 42
    N_THREADS = 4
    N_FOLDS = 5
    TARGET_NAME = "churn"
    TIMEOUT = 300
    MODEL_PATH = "models/automl_model.pkl"

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Set random seeds
    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(N_THREADS)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv("data/raw/student_data.csv")
    train_df, test_df, ohe = preprocess_data(df)

    # Apply SMOTE
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = train_df.drop(columns=["churn"]), train_df["churn"]
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    train_data = pd.concat([X_train_resampled, y_train_resampled], axis=1)

    # Setup LightAutoML
    print("Setting up and training LightAutoML model...")
    task = Task("binary", metric="auc")
    roles = {"target": TARGET_NAME, "drop": ["student_id"]}

    automl = TabularAutoML(
        task=task,
        timeout=TIMEOUT,
        cpu_limit=N_THREADS,
        reader_params={
            "n_jobs": N_THREADS,
            "cv": N_FOLDS,
            "random_state": RANDOM_STATE,
        },
    )

    # Train model
    out_of_fold_predictions = automl.fit_predict(train_data, roles=roles, verbose=1)
    test_predictions = automl.predict(test_df)

    # Calculate and print metrics
    y_test = test_df[TARGET_NAME]
    y_pred = (test_predictions.data[:, 0] > 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\nModel Performance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Save the model
    print(f"\nSaving model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(automl, f)
    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()
