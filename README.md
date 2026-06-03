# Biking Infrastructure Maintenance Projection

## Project Overview
Biking culture is pertinent to life in Flanders, making the maintenance of cycling infrastructure absolutely critical. For the AWV (Agentschap Wegen en Verkeer), identifying which routes require urgent intervention is a constant challenge, especially given Flanders' volatile weather. These maintenance operations must not disrupt the public, and roadworks should ideally be restricted to low-traffic periods. 

This project delivers an end-to-end forecasting solution that predicts which sites urgently require maintenance and identifies the optimal windows for execution over the next seven days. Additionally, we built a centralized monitoring dashboard for AWV MLOps engineers to track pipeline health, data integrity, and predictive performance at a glance.

## Tech Stack & Architecture
Our data pipeline architecture is built for scalability, utilizing robust MLOps practices:
* **Deployment & Containerization:** Deployed on `Amazon EC2`. The environment runs 7 services in separate containers using Docker, communicating via a bridge network with persistent data storage using volumes.
* **Workflow Orchestration:** `Apache Airflow` schedules the pipeline to run daily at 7 AM. It enforces strict task dependencies (Extraction > Forecasting) and handles retries for failed tasks after 5 minutes.
* **Model Tracking:** `MLflow` is integrated for tracking models and logging evaluation metrics (MAE and RMSE). The current production model is actively aliased as `@champion`.
* **User Interface:** A `Streamlit dashboard` consumes the API to display insights for business users and monitoring metrics for engineers.

<p align = 'center'>
  <img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/53f65c23-188f-446e-83be-99315546bf00" />
</p>

## Dataset Overview

AWV makes this data publicly available here:

- Dataset catalogue: https://www.vlaanderen.be/datavindplaats/catalogus/fietstellingen-awv
- Direct CSV files: https://opendata.apps.mow.vlaanderen.be/fietstellingen/index.html

## Data Extraction & Forecasting Pipeline
- **Data Extraction:** Traffic data is collected by performing web scraping from the AWV portal using the `requests` library.
- **Data Storage:** Processed data is sent to an SQLite database. The pipeline dynamically checks if tables contain data, appends new data accordingly, and utilizes unique indexes to avoid duplicity.
- **Model Calibration:** We use a LightGBM model trained on 365 days of historical data and tested on a 30-day split. Inputs include location, data features, and recent daily bike counts.
- **Forecasting:** The model generates future forecasts recursively using previously predicted values. 

## Deployment Guide
To deploy this project on an AWS EC2 instance:
   ```bash
    1. ssh -i my-key-pair-2.pem ec2-user@13.222.164.59 ## SSH into the instance
    2. docker build . -t automated-forecaster ## Build the Docker image
    3. docker-compose up -d ## Spin up the services
   ```
## Dashboard Features

The Streamlit application serves two primary functions:
- **For MLOps Engineers:** Monitors test cases including data completeness checks and model error tracking.
- **For Business Users:** Provides site-specific cycling usage predictions (e.g., searching by Location_Tag_61). It features automated alerts for sites requiring maintenance or targeted snow cleanup over the coming week.

<p align = 'center'>
  <img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/9e1f5955-b39c-4e5a-8900-a786e14f1c9b" />
</p>
