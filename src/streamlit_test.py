#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import requests
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from pathlib import Path
import sqlite3
import openmeteo_requests
import requests_cache
from retry import retry
import pandas as pd
import numpy as np
from datetime import date, datetime
import datetime as dt
import streamlit as st


# In[ ]:


# SERVER_URL = "http://13.222.164.59:5000"
# mlflow.set_tracking_uri(SERVER_URL)
# client = MlflowClient()

# try:
#     print("Searching for the latest run")
#     # Get the experiment metadata
#     experiment = client.get_experiment_by_name("Forecasting Experiment")

#     if experiment is None:
#         raise ValueError("Could not find an experiment named 'Forecasting Experiment'.")

#     # Search for latest successful run
#     runs = client.search_runs(
#         experiment_ids=[experiment.experiment_id],
#         max_results=1,
#         filter_string="attributes.status = 'FINISHED'",
#         order_by=["attributes.start_time DESC"]
#     )

#     if not runs:
#         raise ValueError("No successful runs found in this experiment.")

#     #Extract the true active Run ID
#     latest_run_id = runs[0].info.run_id
#     print(f"🎯 Found latest active Run ID: {latest_run_id}")

#     #Build the precise direct download URL
#     download_url = f"{SERVER_URL}/get-artifact?run_id={latest_run_id}&path=model.pkl"
#     local_filename = "model.pkl"

#     print("Downloading model.pkl")
#     with requests.get(download_url, stream=True) as response:
#         if response.status_code == 200:
#             with open(local_filename, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#             print(f"✅ Successfully downloaded raw file to: {os.path.abspath(local_filename)}")

#             with open(local_filename, "rb") as f:
#                 model = pickle.load(f)
#             print("Model downloaded successfully")
#             print(f"Model Details: {model}")

#         else:
#             print(f"Download failed. Status code: {response.status_code}")
#             print("Response:", response.text)

# except Exception as e:
#     print("An error occurred:")
#     print(e)


# In[2]:


with open('model.pkl','rb') as file:
    model= pickle.load(file)

model


# In[3]:


DB_PATH = "model_metrics.db"
conn = sqlite3.connect(DB_PATH)


# In[8]:


sites_df= pd.read_excel('Data/sites.xlsx')
sites_df.head()


# In[9]:


df_coords =sites_df[['lat','long']]
df_coords['lat']= df_coords['lat'].astype(str)
df_coords['long']= df_coords['long'].astype(str)
df_coords['coord']= df_coords[['lat','long']].agg(', '.join, axis=1)
#df_coords.to_csv('df_coords.csv', index=False)
df_coords.head()


# In[11]:


cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
openmeteo = openmeteo_requests.Client(session=cache_session)


# In[12]:


# 4. Configure Open-Meteo parameters specifically for DAILY snowfall
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": df_coords["lat"].tolist(),
    "longitude": df_coords["long"].tolist(),
    "daily": "snowfall_sum",  # Requests the daily total snowfall sum
    "forecast_days": 7
}

print(f"🛰️ Requesting daily forecast data for the unique locations...")
responses = openmeteo.weather_api(url, params=params)

# Extract arrays of the actual latitudes and longitudes returned by the API for mapping
api_lats = np.array([res.Latitude() for res in responses])
api_ons = np.array([res.Longitude() for res in responses])

all_frames = []

print(f"📋 Processing {len(df_coords)} rows sequentially...")

# 5. Loop through df_coords rows using iterrows() to get a continuous sequence
for idx, row in df_coords.iterrows():
    # Convert index to 1-based numbering (1 to 150)
    site_no = idx + 1
    lat = float(row['lat'])
    lon = float(row['long'])

    # Find the index of the closest API response using Euclidean distance
    distances = (api_lats - lat)**2 + (api_ons - lon)**2
    closest_idx = np.argmin(distances)

    response = responses[closest_idx]

    # 6. Extract the Daily tracking block
    daily = response.Daily()
    snowfall_daily_data = daily.Variables(0).ValuesAsNumpy()

    # Generate daily spaced time indices (one row per day for 7 days)
    start_time = pd.to_datetime(daily.Time(), unit="s", utc=True)
    end_time = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True)
    freq_interval = pd.Timedelta(seconds=daily.Interval())
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq_interval, inclusive="left")

    # 7. Format Site_ID as 'LOCATION_TAG_1', 'LOCATION_TAG_2', etc.
    site_id_string = f"Location_Tag_{site_no}"

    # Construct the daily DataFrame for this specific row number
    loc_df = pd.DataFrame({
        "Site_ID": site_id_string,
        "Date": time_index,
        "latitude": lat,
        "longitude": lon,
        "daily_snowfall_cm": snowfall_daily_data
    })

    all_frames.append(loc_df)

# 8. Combine all slices together
weather_df = pd.concat(all_frames, ignore_index=True)

# Format dates cleanly to standard 'YYYY-MM-DD'
weather_df['Date'] = pd.to_datetime(weather_df['Date']).dt.strftime('%Y-%m-%d')

print("\n🎉 Daily Forecast Successfully Generated!")
print(f"Total rows in dataframe: {len(weather_df)}")
weather_df.head() 


# In[37]:


weather_df.info()


# In[26]:


# Import your pipeline functions from your step2 script
from step2_lightgbm_2y import (
    load_and_prepare_daily, 
    recursive_forecast_lgbm, 
    get_latest_date_from_db,
    FEATURES,
    DEFAULT_DB_PATH,
    DEFAULT_TABLE,
    METRICS_DB_PATH
)

def get_latest_model_from_experiment(experiment_id: str) -> str:
    print(f"Searching for the latest run in Experiment ID: {experiment_id}...")

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError(f"No runs found in Experiment {experiment_id}")

    latest_run_id = runs.iloc[0]["run_id"]
    mae = runs.iloc[0].get("metrics.mae", "N/A")
    print(f"Found Latest Run ID: {latest_run_id} (MAE: {mae})")

    return f"runs:/{latest_run_id}/lgbm_model"

def generate_forecast_from_experiment(
    # tracking_uri: str = "http://13.222.164.59:5000",
    # experiment_id: str = "2",
    db_path: str = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
    metrics_db: str = METRICS_DB_PATH,
    forecast_days: int = 20               #7
):
    # mlflow.set_tracking_uri(tracking_uri)

    # model_uri = get_latest_model_from_experiment(experiment_id)

    # print(f"Loading model artifacts from {model_uri}...")
    lgbm_model = model#mlflow.lightgbm.load_model(model_uri)

    # 🌟 FIX: Change datetime.now() to pull the true maximum available date from your DB
    latest_historical_date = get_latest_date_from_db(db_path=db_path, table=table)
    cutoff = pd.Timestamp(latest_historical_date)
    print(f"📊 Historical data context baseline found at: {cutoff.strftime('%Y-%m-%d')}")

    # latest_historical_date = datetime.now()#get_latest_date_from_db(db_path=db_path, table=table)
    # cutoff = pd.Timestamp(latest_historical_date)

    future_dates = pd.date_range(start=cutoff + pd.Timedelta(days=1), periods=forecast_days, freq="D")

    df_daily, _ = load_and_prepare_daily(
        db_path=db_path, 
        table=table, 
        cutoff=cutoff.strftime('%Y-%m-%d'), 
        forecast_end=cutoff.strftime('%Y-%m-%d'), 
        days=30
    )
    history = df_daily[df_daily["Start_Time"] <= cutoff][["Site_ID", "Start_Time", "Count"]].copy()

    print("Running recursive forecasting engine...")
    pred_df = recursive_forecast_lgbm(
        model=lgbm_model,
        history=history,
        future_dates=future_dates,
        features=FEATURES
    )

    forecast_output = pred_df[["Site_ID", "Start_Time", "pred"]].copy()
    forecast_output = forecast_output.rename(columns={"Start_Time": "Date", "pred": "predicted_count"})

    # Format layout cleanly
    forecast_output['Date'] = forecast_output['Date'].dt.strftime('%Y-%m-%d')
    forecast_output['generation_time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    # forecast_output['model_run_id'] = model_uri.split('/')[1] 

    # 9. Append predictions to model_metrics.db for your dashboard
    print(f"Appending new predictions to the 'cycling_predictions' table in {metrics_db}...")
    with sqlite3.connect(metrics_db) as conn:
        # Saving directly to the requested table name
        forecast_output.to_sql("cycling_predictions", conn, if_exists="append", index=False)

    print("Projections successfully computed")
    return forecast_output.head()

# if __name__ == "__main__":
#     project_dir = Path(__file__).resolve().parent
#     generate_forecast_from_experiment(
#         tracking_uri="http://13.222.164.59:5000",
#         experiment_id="2",
#         db_path=project_dir / DEFAULT_DB_PATH,
#         metrics_db=project_dir / METRICS_DB_PATH
#     )
generate_forecast_from_experiment(
        # tracking_uri="http://13.222.164.59:5000",
        # experiment_id="2",
        db_path=DEFAULT_DB_PATH,
        metrics_db="model_metrics.db"
    )


# In[31]:


forecast_output= pd.read_sql("SELECT * FROM cycling_predictions", conn)
forecast_output.info()


# In[28]:


# forecast_days: int = 7
# pd.date_range(start=date.today() + pd.Timedelta(days=1), periods=forecast_days, freq="D")


# In[32]:


forecast_output['Date'].unique()


# In[45]:


eval_df['Date'].unique()


# In[51]:


model#.predict()


# In[ ]:


# FEATURES = [
#     "Site_ID",
#     "dayofweek",
#     "is_weekend",
#     "month_sin",
#     "month_cos",
#     "Is_Public_Holiday",
#     "lag_1",
#     "lag_2",
#     "lag_3",
#     "lag_7"
# ]


# In[55]:


weather_df.head()


# In[65]:


weather_df['Date'].unique()


# In[55]:


full_df= pd.merge(forecast_output, weather_df, on=['Site_ID','Date'],how='inner')
full_df.head()


# In[49]:


#Chart w ideal maintenance suggestion day w expected cycling usage and snowfall at the time


# In[56]:


full_df['predicted_count'].describe()


# In[57]:


full_df['Site_ID']= full_df['Site_ID'].str.split(r'_').str[-1]
full_df['Maintenance_Snow'] = (full_df['daily_snowfall_cm']>0) & (full_df['predicted_count']< full_df['predicted_count'].quantile(0.10))
full_df['Maintenance_Possibility'] = (full_df['predicted_count']< full_df['predicted_count'].quantile(0.10))
full_df.head()


# In[52]:





# In[54]:


#Dashboard


# In[ ]:


st.title("Model Monitoring Dashboard")
st.write("This dashboard automatically checks for new model outputs every 30 minutes.")

DB_PATH = "model_metrics.db"

@st.fragment(run_every="30m")
def render_dashboard_data():
    try:
        conn = sqlite3.connect(DB_PATH)

        #Site-wise Missing Data Check
        eval_df = pd.read_sql("""SELECT * FROM model_predictions_eval 
                        WHERE timestamp = (SELECT MAX(timestamp)
                                            FROM model_predictions_eval)
                                            ORDER BY Site_ID""", conn)
        eval_df.rename(columns={'Start_Time':'Date','pred':'Predicted_Values','actual':'Actual_Values'}, inplace=True)
        eval_df['Date']= pd.to_datetime(eval_df['Date']).dt.strftime('%Y-%m-%d')
        eval_df['timestamp']= pd.to_datetime(eval_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        unique_dates = eval_df['Date'].nunique()
        unique_sites = eval_df['Site_ID'].nunique()
        expected_rows = unique_dates * 151
        actual_rows = len(eval_df)

        st.write("Were cycling usage forecasts made for all dates-sites combinations?")
        if actual_rows == expected_rows and unique_sites == 151:
            st.success("Every date has a cycling usage prediction for all 151 sites.")
        elif actual_rows <= expected_rows:
            st.error("There are less than expected date-site combinations.")
        else:
            st.error("There are more than expected date-site combinations, possible duplicates.")

        #Weather API hit: successful or not

        # missing_data = eval_df.isnull().sum()
        # # st.subheader("Were cycling usage predictions generated for all sites?")
        # if missing_data.sum() > 0:
        #     st.error(f":red[Alert: Missing predictions detected for {missing_data[missing_data > 0].index.tolist()}]")
        # else:
        #     st.success("Prediction generated for all sites")

        #Plot error metrics (MAE/RMSE)
        st.subheader("Model Accuracy (MAE/RMSE)")
        health_df = pd.read_sql("SELECT timestamp, mae as MAE, rmse as RMSE FROM pipeline_health ORDER BY timestamp", conn)
        health_df['timestamp']= pd.to_datetime(health_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.line_chart(health_df.set_index('timestamp')[['MAE', 'RMSE']], color=['red','orange'])

        #Cycling Usage: Predictions vs Actuals
        st.subheader("Latest Cycling Usage Predictions")
        st.caption(f"Last prediction time: {pd.to_datetime(eval_df['timestamp']).max()}")
        st.write("Check Cycling usage predictions by entering Site ID below, eg.: Location_Tag_10 ")
        siteID= st.text_input("Enter SiteID")
        if siteID:
            st.line_chart(eval_df[eval_df['Site_ID']==siteID].set_index('Date')[['Predicted_Values', 'Actual_Values']], color=['blue','green'])

        #Maintenance Timing Prediction
        st.write("Were maintenance predictions made for all dates-sites combinations?")
        unique_dates = full_df['Date'].nunique()
        unique_sites = full_df['Site_ID'].nunique()
        expected_rows = unique_dates * 152
        actual_rows = len(full_df)
        if actual_rows == expected_rows and unique_sites == 152:
            st.success("Every date has a maintenance prediction for all 151 sites.")
        elif actual_rows <= expected_rows:
            st.error("There are less than expected date-site combinations.")
        else:
            st.error("There are more than expected date-site combinations, possible duplicates.")

        st.subheader("Prediction for Site Maintenance in Next 7 Days")
        st.caption(f"Last prediction time: {pd.to_datetime(full_df['generation_time']).max()}")

        sites_to_maintain = full_df[full_df['Maintenance_Possibility']]['Site_ID'].unique()
        st.markdown(f":red[Alert: Sites to be maintained this week are {sites_to_maintain}]")
        #st.write("Check Maintenance Projection by entering Site ID below, eg.: 'Location_Tag_1'")
        siteID3= st.text_input("Check Maintenance Projection by entering Site ID below, eg.: 10 ")
        if siteID3:
            #plot of date-wise maintenace yes/no (coloured box/column or something)
            st.line_chart(full_df[full_df['Site_ID']==siteID3].set_index('Date')[['Maintenance_Possibility']])

        sites_for_snow = full_df[full_df['Maintenance_Snow']]['Site_ID'].unique()
        st.markdown(f":red[Alert: Sites to focus on for snow cleanup this week are {sites_for_snow}]")
        #st.write("Check Snow Clean-up Projection by entering Site ID below, eg.: 'Location_Tag_1'")
        siteID2= st.text_input("Check Snow Clean-up Projection by entering Site ID below, eg.: 10 ")
        if siteID2:
            #plot of date-wise maintenace yes/no (coloured box/column or something)
            st.line_chart(full_df[full_df['Site_ID']==siteID2].set_index('Date')[['Maintenance_Snow']])   
        conn.close()

    except Exception as e:
        st.error(f"Error loading data: {e}")

render_dashboard_data()


# In[ ]:


st.sidebar.markdown("[Go to MLflow Server](http://13.222.164.59:5000/#/experiments/2/runs)")
st.sidebar.markdown("[Go to Airflow Server](http://13.222.164.59:8080/home)") 

