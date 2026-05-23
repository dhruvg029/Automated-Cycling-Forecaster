import pandas as pd
from fastapi import FastAPI
import mlflow.pyfunc
from src.step2_lightgbm_2y import load_and_prepare_daily, recursive_forecast_lgbm

import pickle




app = FastAPI()

mlflow.set_tracking_uri("http://mlflow:5000")


@app.get("/")
def home():
    return {"message": "home"}


@app.get("/getForecasts")
def getForecasts(loc, start, end):

    print(f"loc={loc}, start={start}, end={end}")

    model = mlflow.pyfunc.load_model("models:/forecaster@champion")

    df_daily, _ = load_and_prepare_daily(cutoff=start, forecast_end=end, days=15)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    history = df_daily[(df_daily["Start_Time"] <= start_ts) & (df_daily["Site_ID"] == f"Location_Tag_{str(loc)}")][["Site_ID", "Start_Time", "Count"]].copy()
    future_dates = pd.date_range(start=start_ts, end=end_ts, freq="D")

    forecasts = recursive_forecast_lgbm(model=model, history=history, future_dates=future_dates)
    forecasts["date"] = forecasts["Start_Time"].dt.date.astype(str)
    forecasts["forecast"] = forecasts["pred"].astype(int)
    forecasts_out = forecasts[["date", "forecast"]].set_index("date")["forecast"].to_dict()

    out = {
        "loc": f"Location_Tag_{str(loc)}",
        "start": str(start),
        "end": str(end),
        'forecasts': forecasts_out
    }
    
    return out
