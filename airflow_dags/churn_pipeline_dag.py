from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
}

def run_preprocessing():
    os.system("python src/preprocessing.py")

def run_training():
    os.system("python src/train.py")

def run_evaluation():
    os.system("python src/evaluate.py")

with DAG(
    "churn_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable=run_preprocessing,
    )

    training = PythonOperator(
        task_id="training",
        python_callable=run_training,
    )

    evaluation = PythonOperator(
        task_id="evaluation",
        python_callable=run_evaluation,
    )

    preprocessing >> training >> evaluation