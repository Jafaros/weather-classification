import numpy as np
import pandas as pd
from pandas import DataFrame

TARGET = "RainTomorrow"
NUMERIC_FEATURES = [
    "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
]
WIND_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

# Funkce pro normalizaci dat
def normalize_data(df: DataFrame) -> DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    year_angle = 2 * np.pi * (dates.dt.dayofyear - 1) / np.where(
        dates.dt.is_leap_year, 366, 365
    )
    wind_angle = np.deg2rad(df["WindGustDir"].map(
        {direction: index * 22.5 for index, direction in enumerate(WIND_DIRECTIONS)}
    ))

    # Cyklické veličiny zůstávají blízké i přes hranici roku nebo 0°/360°.
    df["DayOfYearSin"] = np.sin(year_angle)
    df["DayOfYearCos"] = np.cos(year_angle)
    df["WindGustDirSin"] = np.sin(wind_angle)
    df["WindGustDirCos"] = np.cos(wind_angle)
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return remove_columns(df, ["Date", "WindGustDir"]).replace(
        [np.inf, -np.inf], np.nan
    )


# Funkce pro odstranění sloupců
def remove_columns(df: DataFrame, columns_to_remove: list) -> DataFrame:
    return df.drop(columns=columns_to_remove)


# Funkce pro odstranění řádků s chybějícími hodnotami
def remove_null_rows(df: DataFrame) -> DataFrame:
    return df.dropna()


# Funkce pro předzpracování dat
def preprocess_data(df: DataFrame) -> DataFrame:
    selected = df[NUMERIC_FEATURES + ["Date", "WindGustDir", TARGET]]
    result = remove_null_rows(normalize_data(selected))
    result[TARGET] = result[TARGET].astype(int)
    
    return result.reset_index(drop=True)
