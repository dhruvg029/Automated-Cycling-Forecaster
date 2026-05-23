#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import requests
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import streamlit as st


# In[ ]:


SERVER_URL = "http://13.222.164.59:5000"
mlflow.set_tracking_uri(SERVER_URL)
client = MlflowClient()

try:
    print("Searching for the latest run")
    # Get the experiment metadata
    experiment = client.get_experiment_by_name("Forecasting Experiment")

    if experiment is None:
        raise ValueError("Could not find an experiment named 'Forecasting Experiment'.")

    # Search for latest successful run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"]
    )

    if not runs:
        raise ValueError("No successful runs found in this experiment.")

    #Extract the true active Run ID
    latest_run_id = runs[0].info.run_id
    print(f"🎯 Found latest active Run ID: {latest_run_id}")

    #Build the precise direct download URL
    download_url = f"{SERVER_URL}/get-artifact?run_id={latest_run_id}&path=model.pkl"
    local_filename = "model.pkl"

    print("Downloading model.pkl")
    with requests.get(download_url, stream=True) as response:
        if response.status_code == 200:
            with open(local_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Successfully downloaded raw file to: {os.path.abspath(local_filename)}")

            with open(local_filename, "rb") as f:
                model = pickle.load(f)
            print("Model downloaded successfully")
            print(f"Model Details: {model}")

        else:
            print(f"Download failed. Status code: {response.status_code}")
            print("Response:", response.text)

except Exception as e:
    print("An error occurred:")
    print(e)


# In[ ]:


st.title("Scheduled Model Monitoring Dashboard")
st.write("This dashboard automatically checks for new model outputs every 30 minutes.")

DB_PATH = "model_metrics.db"

@st.fragment(run_every="30m")
def render_dashboard_data():
    try:
        conn = sqlite3.connect(DB_PATH)

        #Site-wise Missing Data Check
        eval_df = pd.read_sql("SELECT * FROM model_predictions_eval ORDER BY timestamp DESC LIMIT 150", conn)
        missing_data = eval_df.isnull().sum()
        st.subheader("Were predictions generated for all sites?")
        if missing_data.sum() > 0:
            st.error(f"Alert: Missing predictions detected for {missing_data[missing_data > 0].index.tolist()}")
        else:
            st.success("Prediction generated for all sites.")

        #Plot health (MAE/RMSE)
        st.subheader("Model Accuracy (MAE/RMSE)")
        health_df = pd.read_sql("SELECT * FROM pipeline_health ORDER BY timestamp DESC", conn)
        st.line_chart(health_df.set_index('timestamp')[['mae', 'rmse']])

        #Predictions vs Actuals Table
        st.subheader("Latest Predictions")
        st.dataframe(eval_df)
        st.caption(f"Last prediction time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        conn.close()

    except Exception as e:
        st.error(f"Error loading data: {e}")

render_dashboard_data()


# In[ ]:


st.sidebar.markdown("[Go to MLflow Server](http://13.222.164.59:5000)")


# In[ ]:




