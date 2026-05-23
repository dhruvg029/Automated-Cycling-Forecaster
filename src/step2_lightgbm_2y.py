"""
LightGBM recursive forecast pipeline (1-year training window).

Ported from step2_LightGBM.ipynb — 1Y path only (train → lgbm_model → eval_df).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays

## Dhruv => Added MLflow imports
import mlflow
import mlflow.lightgbm

from argparse import ArgumentParser

FEATURES = [
    "Site_ID",
    "dayofweek",
    "is_weekend",
    "month_sin",
    "month_cos",
    "Is_Public_Holiday",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7"
]
TARGET = "Count"
DEFAULT_DB_PATH = "/data/fietstellingen.db"
DEFAULT_OUT_PATH = "/data/eval_df.csv"
DEFAULT_TABLE = "traffic_counts"


def load_raw_data(
    db_path: str | Path = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
    cutoff: str | pd.Timestamp = "2026-03-31",
    forecast_end: str | pd.Timestamp = "2026-04-30",
    days: int = 365*2,
) -> pd.DataFrame:
    """Load cycling counts from SQLite (step2_LightGBM.ipynb)."""

    max_lag = 7
    start_time = (pd.Timestamp(cutoff) - pd.Timedelta(days=days + max_lag)).strftime("%Y-%m-%d") + " 00:00:00"
    end_time = pd.Timestamp(forecast_end).strftime("%Y-%m-%d") + " 23:59:59"

    query = f"""
    SELECT 
        Site_ID,
        strftime('%Y-%m-%d %H:00:00', Start_Time) AS Start_Time,
        SUM(Count) AS Count 
    FROM "{table}"
    WHERE Start_Time >= "{start_time}"
    AND Start_Time <= "{end_time}"
    GROUP BY
        Site_ID,
        strftime('%Y-%m-%d %H:00:00', Start_Time)
    ORDER BY
        Site_ID,
        Start_Time;
    """

    print(f"query: {query}")
    with sqlite3.connect(Path(db_path)) as conn:
        return pd.read_sql_query(query, conn)


def load_and_prepare_daily(
    db_path: str | Path = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
    cutoff: str | pd.Timestamp = "2026-03-31",
    forecast_end: str | pd.Timestamp = "2026-04-30",
    days: int = 365*2,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    """Load raw counts, aggregate to daily per site, add features and lags."""

    df_h = load_raw_data(db_path, table, cutoff, forecast_end, days)
    df_h["Start_Time"] = pd.to_datetime(df_h["Start_Time"])

    df_daily = (
        df_h.dropna(subset=["Count"])
        .set_index("Start_Time")
        .groupby("Site_ID")
        .resample("D")
        .agg({"Count": "sum"})
        .reset_index()
    )
    
    df_daily["Start_Time"] = pd.to_datetime(df_daily["Start_Time"])

    df_daily = add_time_features(df_daily)
    df_daily = add_holiday_feature(df_daily)

    for lag in (1, 2, 3, 7):
        df_daily[f"lag_{lag}"] = df_daily.groupby("Site_ID")["Count"].shift(lag)

    df_model = df_daily.dropna().copy()


    return df_daily, df_model


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["dayofweek"] = data["Start_Time"].dt.dayofweek
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
    data["month"] = data["Start_Time"].dt.month
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    return data

def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    ## Initialize the Belgian holiday calendar for your dataset's years
    years_in_data = df['Start_Time'].dt.year.unique().tolist()
    be_holidays = holidays.BE(years=years_in_data)
    ## Create a binary flag: 1 if it's a holiday, 0 if it's a normal day
    ## We convert Start_Time to just the date to match the holiday dictionary
    df['Is_Public_Holiday'] = df['Start_Time'].dt.date.apply(lambda x: 1 if x in be_holidays else 0)
    return df


def split_train_test(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp = "2026-03-31",
    forecast_end: str | pd.Timestamp = "2026-04-30",
    days: int = 365*2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(cutoff)
    forecast_end = pd.Timestamp(forecast_end)
    train_start = cutoff - pd.DateOffset(days=days)

    train = df[
        (df["Start_Time"] >= train_start) & (df["Start_Time"] <= cutoff)
    ].copy()
    test_actual = df[
        (df["Start_Time"] > cutoff) & (df["Start_Time"] <= forecast_end)
    ].copy()
    return train, test_actual

## Dhruv => added model_params argument for MLFlow
def fit_lgbm(train: pd.DataFrame, model_params: dict) -> LGBMRegressor:
    train = train.copy()
    train["Site_ID"] = train["Site_ID"].astype("category")

    ## Dhruv => Model parameters defined here
    model = LGBMRegressor(**model_params)

    model.fit(
        train[FEATURES],
        train[TARGET],
        categorical_feature=["Site_ID"],
    )
    return model


def recursive_forecast_lgbm(
    model: LGBMRegressor,
    history: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    history = history.copy()
    predictions = []
    sites = history["Site_ID"].unique()

    for date in future_dates:
        future_rows = pd.DataFrame({"Site_ID": sites, "Start_Time": date})
        future_rows = add_time_features(future_rows)
        future_rows = add_holiday_feature(future_rows)

        for lag in (1, 2, 3, 7):
            lag_values = (
                history[history["Start_Time"] == date - pd.Timedelta(days=lag)][
                    ["Site_ID", "Count"]
                ].rename(columns={"Count": f"lag_{lag}"})
            )
            future_rows = future_rows.merge(lag_values, on="Site_ID", how="left")

        future_rows = future_rows.dropna(
            subset=["lag_1", "lag_2", "lag_3", "lag_7"]
        ).copy()
        future_rows["Site_ID"] = future_rows["Site_ID"].astype("category")
        future_rows["pred"] = model.predict(future_rows[features])
        future_rows["pred"] = np.maximum(future_rows["pred"], 0)

        append_rows = future_rows[["Site_ID", "Start_Time", "pred"]].rename(
            columns={"pred": "Count"}
        )
        history = pd.concat([history, append_rows], ignore_index=True)
        predictions.append(future_rows)

    return pd.concat(predictions, ignore_index=True)


def predict_and_evaluate(
    lgbm_model: LGBMRegressor,
    test_actual: pd.DataFrame,
    df_daily: pd.DataFrame,
    cutoff: str | pd.Timestamp = "2026-03-31",
    forecast_end: str | pd.Timestamp = "2026-04-30",
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    forecast_end = pd.Timestamp(forecast_end)

    future_dates = pd.date_range(
        start=cutoff + pd.Timedelta(days=1),
        end=forecast_end,
        freq="D",
    )
    history = df_daily[df_daily["Start_Time"] <= cutoff][
        ["Site_ID", "Start_Time", "Count"]
    ].copy()

    pred_eval = recursive_forecast_lgbm(
        model=lgbm_model,
        history=history,
        future_dates=future_dates,
        features=features,
    )

    eval_df = pred_eval.merge(
        test_actual[["Site_ID", "Start_Time", "Count"]],
        on=["Site_ID", "Start_Time"],
        how="inner",
    )
    eval_df = eval_df.rename(columns={"Count": "actual"})

    mae = mean_absolute_error(eval_df["actual"], eval_df["pred"])
    rmse = np.sqrt(mean_squared_error(eval_df["actual"], eval_df["pred"]))
    print("Recursive LightGBM MAE:", mae)
    print("Recursive LightGBM RMSE:", rmse)

    ## Dhruv => Added more metrics here for MLflow to track
    return eval_df, mae, rmse

def run_pipeline(
    db_path: str | Path = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
    cutoff: str = "2026-03-31",
    forecast_end: str = "2026-04-30",
    train_days: int = 365*2,
) -> dict:

    """Full 1Y pipeline: load → features → split → fit → recursive forecast → eval_df."""

    df_daily, df_model = load_and_prepare_daily(db_path, table, cutoff, forecast_end, train_days)
    train, test_actual = split_train_test(
        df_model,
        cutoff=cutoff,
        forecast_end=forecast_end,
        days=train_days,
    )

    train = train.sort_values(["Site_ID", "Start_Time"]).reset_index(drop=True)
    test_actual = test_actual.sort_values(["Site_ID", "Start_Time"]).reset_index(drop=True)

    ## Dhruv => Setting experiment name
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Forecasting Experiment")

    ## Dhruv => Added model parameters here instead
    lgbm_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": -1,
        "num_leaves": 31,
        "random_state": 42,
    }

    ## Dhruv => Starting MLflow run here
    with mlflow.start_run(run_name = "Dhruv_Testing_1"):

        ## Dhruv => Logging the parameters and config
        mlflow.log_params(lgbm_params)
        mlflow.log_param("train_days", train_days)
        mlflow.log_param("cutoff_date", cutoff)

        lgbm_model = fit_lgbm(train, lgbm_params)
        
        ## Updated here the function
        eval_df, mae, rmse= predict_and_evaluate(
            lgbm_model,
            test_actual,
            df_daily,
            cutoff = cutoff,
            forecast_end = forecast_end,
        )

        ## Dhruv => Logging the final metrics
        mlflow.log_metric("Mean_Absolute_Error", mae)
        mlflow.log_metric("Root_Mean_Squared_Error", rmse)
        
        ## Dhruv => Saving the actual model artifact to MLflow
        ## Dhruv => This is what docker will pull later to serve the predictions
        mlflow.lightgbm.log_model(lgbm_model, "model")

    return {
        "df_daily": df_daily,
        "df_model": df_model,
        "train": train,
        "test_actual": test_actual,
        "lgbm_model": lgbm_model,
        "eval_df": eval_df,
    }


def main() -> None:
    # parse arguments from CLI
    parser = ArgumentParser()
    parser.add_argument("--cutoff", type=str, default="2026-03-31")
    parser.add_argument("--forecast_end", type=str, default="2026-04-30")
    parser.add_argument("--train_days", type=int, default=365)
    args = parser.parse_args()

    print(f"Set params: cutoff={args.cutoff}, forecast_end={args.forecast_end}, train_days={args.train_days}")

    project_dir = Path(__file__).resolve().parent
    db_path = project_dir / DEFAULT_DB_PATH
    out_path = project_dir / DEFAULT_OUT_PATH

    results = run_pipeline(db_path=db_path, cutoff=args.cutoff, forecast_end=args.forecast_end, train_days=args.train_days)
    results["eval_df"].to_csv(out_path, index=False)
    print(f"Saved eval_df to {out_path}")


if __name__ == "__main__":
    main()