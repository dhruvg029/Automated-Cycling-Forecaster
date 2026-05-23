from datetime import datetime, timedelta
from airflow import DAG
# from airflow.providers.standard.operators.bash import BashOperator
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

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

    # extract_data = BashOperator(
    #     task_id="extract_data",
    #     bash_command="docker-compose run data-collector",
    #     # bash_command="python src/step1_data_extraction.py",
    #     # cwd="/Users/rinyoshida/Downloads/KUL/2025_2026/MDA/project/mda-cycling",
    #     # cwd="/app"
    # )

    extract_data = DockerOperator(
        task_id="extract_data",
        image="mda-cycling",
        command="python app/src/step1_data_extraction.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="mda-net",
        auto_remove=True,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source="mda-cycling_mda",
                target="/data",
                type="volume",
            )
        ],
    )

    # run_lightgbm_pipeline = BashOperator(
    #     task_id="run_lightgbm_pipeline",
    #     bash_command="docker-compose run model-calibrator",
    #     # bash_command="python src/step2_lightgbm_2y.py",
    #     # cwd="/Users/rinyoshida/Downloads/KUL/2025_2026/MDA/project/mda-cycling",
    #     # cwd="/app"
    # )

    run_lightgbm_pipeline = DockerOperator(
        task_id="run_lightgbm_pipeline",
        image="mda-cycling",
        command="python app/src/step2_lightgbm_2y.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="mda-net",
        auto_remove=True,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source="mda-cycling_mda",
                target="/data",
                type="volume",
            )
        ],
    )

    extract_data >> run_lightgbm_pipeline