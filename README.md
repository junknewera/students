Student Churn Prediction
This project predicts student churn in an online master's program using a machine learning model. It includes data generation, model training, inference, and orchestration with Airflow and Docker.
Project Structure

data/raw/: Raw synthetic data (student_data.csv).
data/processed/: Model predictions (predictions.csv).
scripts/: Python scripts for data generation, training, and inference.
airflow/dags/: Airflow DAGs for scheduling inference.
Dockerfile: Docker container for model inference.
docker-compose.yml: Orchestration for Airflow and inference services.

Setup

Clone the repository:git clone git@github.com:junknewera/students.git
cd students


Create a virtual environment and install dependencies:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Generate synthetic data:python scripts/generate_data.py



Next Steps

Train the model: python scripts/train_model.py
Run inference: python scripts/predict.py
Deploy with Docker and Airflow (see docker-compose.yml).

Requirements

Python 3.9+
Docker
Docker Compose
Airflow (optional, for orchestration)

