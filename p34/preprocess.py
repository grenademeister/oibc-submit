#!/usr/bin/env python3
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pvlib, pytz

# ======================
# Constants
# ======================
TARGET = "nins"
CAT_FEATURES = ["pv_id", "cluster"]
EXCLUDE_FEATURES = {"time", "type", "energy", TARGET}
TIMEZONE = "Asia/Seoul"
DEFAULT_LATITUDE = 36.0
DEFAULT_LONGITUDE = 128.0


# ======================
# Data Loading
# ======================
def load_dataset(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load a dataset CSV and localize time zone if 'time' column exists."""
    logging.info(f"Loading dataset from {path}...")
    df = pd.read_csv(path, nrows=nrows)
    logging.info(f"  Shape: {df.shape}")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize(TIMEZONE)

    return df


# ======================
# Feature Engineering
# ======================
def engineer_features(
    df: pd.DataFrame, latitude: float = DEFAULT_LATITUDE, longitude: float = DEFAULT_LONGITUDE
) -> pd.DataFrame:
    """Add time, solar, and weather-based engineered features."""
    logging.info("Engineering features...")
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize(TIMEZONE)

    original_index = df.index
    sort_keys = ["pv_id", "time"] if "pv_id" in df.columns else ["time"]
    df = df.sort_values(sort_keys)

    df = add_temporal_features(df)
    df = add_solar_features(df, latitude, longitude)
    df = add_weather_interactions(df)
    df = interpolate_weather(df)

    df = df.loc[original_index].reset_index(drop=True)
    logging.info(f"  Total features after engineering: {len(df.columns)}")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    minute_of_day = df["time"].dt.hour * 60 + df["time"].dt.minute
    df["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    df["day_of_year"] = df["time"].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["week_of_year"] = df["time"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["time"].dt.weekday >= 5).astype(np.int8)
    df["month"] = df["time"].dt.month.astype(np.int8)
    df["season"] = ((df["month"] % 12 + 3) // 3).astype(np.int8)
    df["minute_of_day"] = minute_of_day.astype(np.int16)
    return df


def add_solar_features(
    df: pd.DataFrame, latitude: float, longitude: float
) -> pd.DataFrame:
    times = df["time"]
    if times.dt.tz is None:
        times = times.dt.tz_localize(TIMEZONE)
    solar = pvlib.solarposition.get_solarposition(
        times.dt.tz_convert(pytz.utc), latitude=latitude, longitude=longitude
    )
    # Use .values to avoid index alignment issues
    df["solar_altitude"] = solar["apparent_elevation"].clip(lower=0).values
    df["solar_azimuth"] = solar["azimuth"].values
    df["sun_up"] = (df["solar_altitude"] > 0).astype(np.int8)
    df["solar_altitude_norm"] = df["solar_altitude"] / 90.0
    return df


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if {"temp_max", "temp_min"}.issubset(df.columns):
        df["temp_range"] = df["temp_max"] - df["temp_min"]
    if {"temp_a", "temp_b"}.issubset(df.columns):
        df["temp_diff_surface"] = df["temp_a"] - df["temp_b"]
    if {"pressure", "ground_press"}.issubset(df.columns):
        df["pressure_gap"] = df["pressure"] - df["ground_press"]
    if {"real_feel_temp", "real_feel_temp_shade"}.issubset(df.columns):
        df["feels_gap"] = df["real_feel_temp"] - df["real_feel_temp_shade"]
    if {"wind_spd_a", "wind_spd_b"}.issubset(df.columns):
        df["wind_resultant"] = np.hypot(
            df["wind_spd_a"].fillna(0), df["wind_spd_b"].fillna(0)
        )
    if "wind_gust_spd" in df.columns:
        df["gust_ratio"] = df["wind_gust_spd"] / (df["wind_resultant"] + 1e-3)
    if {"cloud_a", "cloud_b"}.issubset(df.columns):
        df["cloud_cover_mean"] = df[["cloud_a", "cloud_b"]].mean(axis=1)
    if {"temp_a", "dew_point"}.issubset(df.columns):
        df["dewpoint_spread"] = df["temp_a"] - df["dew_point"]
    if {"humidity", "rel_hum"}.issubset(df.columns):
        rel = df["rel_hum"].replace(0, np.nan)
        df["humidity_ratio"] = df["humidity"] / rel
    if "precip_1h" in df.columns:
        df["is_rainy"] = (df["precip_1h"].fillna(0) > 0).astype(np.int8)
    return df


def interpolate_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Forward/back fill numeric features grouped by pv_id."""
    weather_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_FEATURES and c not in CAT_FEATURES
    ]
    if "pv_id" in df.columns and weather_cols:
        df[weather_cols] = df.groupby("pv_id")[weather_cols].bfill().ffill()
    return df


# ======================
# Feature Alignment & Matrix
# ======================
def align_feature_frames(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cat_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align train/test numeric and categorical features safely."""
    train_filled = train_df.copy()
    test_filled = test_df.copy()

    common_numeric = [
        c for c in train_filled.select_dtypes(include=[np.number]).columns if c in test_filled.columns
    ]
    if common_numeric:
        medians = train_filled[common_numeric].median().fillna(0)
        train_filled[common_numeric] = train_filled[common_numeric].fillna(medians)
        test_filled[common_numeric] = test_filled[common_numeric].fillna(medians)

    for col in cat_features:
        if col not in train_filled.columns or col not in test_filled.columns:
            continue
        train_filled[col] = train_filled[col].astype("category")
        test_filled[col] = test_filled[col].astype("category")
        mode_value = train_filled[col].mode(dropna=True)
        fill_value = mode_value.iloc[0] if not mode_value.empty else "missing"
        if fill_value not in train_filled[col].cat.categories:
            train_filled[col] = train_filled[col].cat.add_categories([fill_value])
        if fill_value not in test_filled[col].cat.categories:
            test_filled[col] = test_filled[col].cat.add_categories([fill_value])
        train_filled[col] = train_filled[col].fillna(fill_value)
        test_filled[col] = test_filled[col].fillna(fill_value)

    return train_filled, test_filled


def build_feature_matrix(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    logging.info("Building feature matrices...")
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_FEATURES]
    cat_features = [c for c in CAT_FEATURES if c in feature_cols]
    train_features = train_df[feature_cols]
    test_features = test_df[[c for c in feature_cols if c in test_df.columns]]
    train_features, test_features = align_feature_frames(train_features, test_features, cat_features)
    return train_features, test_features, feature_cols, cat_features


def encode_for_xgb(df: pd.DataFrame, cat_features: List[str]) -> pd.DataFrame:
    """Encode categorical columns numerically for XGBoost."""
    if not cat_features:
        return df
    for col in cat_features:
        df[col] = df[col].cat.codes.astype("int16")
    return df
