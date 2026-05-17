"""
LightGBM recursive forecast pipeline (2-year training window).

Ported from step2_LightGBM.ipynb — 2Y path only (train_2Y → lgbm_model_2Y → eval_df_2Y).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = [
    "Site_ID",
    "dayofweek",
    "month",
    "dayofyear",
    "is_weekend",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
]
TARGET = "Count"
DEFAULT_DB_PATH = "fietstellingen.db"
DEFAULT_TABLE = "traffic_counts"


def load_raw_data(
    db_path: str | Path = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
) -> pd.DataFrame:
    """Load cycling counts from SQLite (step2_LightGBM.ipynb)."""

    query = f"""
    SELECT *
    FROM "{table}"
    WHERE Start_Time >= datetime('now', '-3 years');
    """

    with sqlite3.connect(Path(db_path)) as conn:
        return pd.read_sql_query(query, conn)


def load_and_prepare_daily(
    db_path: str | Path = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    """Load raw counts, aggregate to daily per site, add features and lags."""
    
    df = load_raw_data(db_path, table)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    df["End_Time"] = pd.to_datetime(df["End_Time"])

    df_daily = (
        df.dropna(subset=["Count"])
        .set_index("Start_Time")
        .groupby("Site_ID")
        .resample("D")
        .agg({"Count": "sum"})
        .reset_index()
    )
    df_daily = add_time_features(df_daily)

    for lag in (1, 7, 14, 28):
        df_daily[f"lag_{lag}"] = df_daily.groupby("Site_ID")["Count"].shift(lag)

    df_model = df_daily.dropna().copy()
    return df_daily, df_model


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["dayofweek"] = data["Start_Time"].dt.dayofweek
    data["month"] = data["Start_Time"].dt.month
    data["dayofyear"] = data["Start_Time"].dt.dayofyear
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["doy_sin"] = np.sin(2 * np.pi * data["dayofyear"] / 365)
    data["doy_cos"] = np.cos(2 * np.pi * data["dayofyear"] / 365)
    return data


def split_train_test(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp = "2025-05-16",
    forecast_end: str | pd.Timestamp = "2025-11-16",
    years: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(cutoff)
    forecast_end = pd.Timestamp(forecast_end)
    train_start = cutoff - pd.DateOffset(years=years)

    train = df[
        (df["Start_Time"] >= train_start) & (df["Start_Time"] <= cutoff)
    ].copy()
    test_actual = df[
        (df["Start_Time"] > cutoff) & (df["Start_Time"] <= forecast_end)
    ].copy()
    return train, test_actual


def fit_lgbm_2y(train_2y: pd.DataFrame) -> LGBMRegressor:
    train_2y = train_2y.copy()
    train_2y["Site_ID"] = train_2y["Site_ID"].astype("category")

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42,
    )
    model.fit(
        train_2y[FEATURES],
        train_2y[TARGET],
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

        for lag in (1, 7, 14, 28):
            lag_values = (
                history[history["Start_Time"] == date - pd.Timedelta(days=lag)][
                    ["Site_ID", "Count"]
                ].rename(columns={"Count": f"lag_{lag}"})
            )
            future_rows = future_rows.merge(lag_values, on="Site_ID", how="left")

        future_rows = future_rows.dropna(
            subset=["lag_1", "lag_7", "lag_14", "lag_28"]
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
    cutoff: str | pd.Timestamp = "2025-05-16",
    forecast_end: str | pd.Timestamp = "2025-11-16",
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

    return eval_df


def run_pipeline(
    db_path: str | Path = DEFAULT_DB_PATH,
    table: str = DEFAULT_TABLE,
    cutoff: str = "2025-05-16",
    forecast_end: str = "2025-11-16",
    train_years: int = 2,
) -> dict:

    """Full 2Y pipeline: load → features → split → fit → recursive forecast → eval_df_2Y."""

    df_daily, df_model = load_and_prepare_daily(db_path, table)
    train_2y, test_actual_2y = split_train_test(
        df_model,
        cutoff=cutoff,
        forecast_end=forecast_end,
        years=train_years,
    )
    lgbm_model_2y = fit_lgbm_2y(train_2y)
    eval_df_2y = predict_and_evaluate(
        lgbm_model_2y,
        test_actual_2y,
        df_daily,
        cutoff=cutoff,
        forecast_end=forecast_end,
    )
    return {
        "df_daily": df_daily,
        "df_model": df_model,
        "train_2Y": train_2y,
        "test_actual_2Y": test_actual_2y,
        "lgbm_model_2Y": lgbm_model_2y,
        "eval_df_2Y": eval_df_2y,
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    db_path = project_dir / DEFAULT_DB_PATH
    out_path = project_dir / "eval_df_2Y.csv"

    results = run_pipeline(db_path=db_path)
    results["eval_df_2Y"].to_csv(out_path, index=False)
    print(f"Saved eval_df_2Y to {out_path}")


if __name__ == "__main__":
    main()
