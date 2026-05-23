from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

default_args = {
    "owner": "rin",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="daily_forecasting_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 5, 21),
    schedule="0 5 * * *",   # UTC 05:00 = Belgium 07:00 (summer)
    catchup=False,
    tags=["forecasting"],
) as dag:

    extract_data = BashOperator(
        task_id="extract_data",
        bash_command="python src/step1_data_extraction.py",
        cwd="/Users/rinyoshida/Downloads/KUL/2025_2026/MDA/project/mda-cycling",
        # cwd="/app"
    )

    run_lightgbm_pipeline = BashOperator(
        task_id="run_lightgbm_pipeline",
        bash_command="python src/step2_lightgbm_2y.py",
        cwd="/Users/rinyoshida/Downloads/KUL/2025_2026/MDA/project/mda-cycling",
        # cwd="/app"
    )

    extract_data >> run_lightgbm_pipeline