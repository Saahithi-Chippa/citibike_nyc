from datetime import datetime, timedelta, timezone
import pytz
import hopsworks
import numpy as np
import pandas as pd
from hsfs.feature_store import FeatureStore

import src.config as config
from src.data_utils import transform_ts_data_info_features


def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
    )


def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    predictions = model.predict(features)

    results = pd.DataFrame()
    results["start_station_id"] = features["start_station_id"].values
    results["predicted_demand"] = predictions.round(0)

    return results


def load_batch_of_features_from_store(
    current_date: datetime,
) -> pd.DataFrame:
    feature_store = get_feature_store()

    # read time-series data from the feature store
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
    )

    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
    )
    ts_data = ts_data[ts_data.start_hour.between(fetch_data_from, fetch_data_to)]

    # Sort data by location and time
    ts_data.sort_values(by=["start_station_id", "start_hour"], inplace=True)

    features = transform_ts_data_info_features(
        ts_data, window_size=24 * 28, step_size=23
    )

    return features


def load_model_from_registry(version=None):
    from pathlib import Path

    import joblib

    from src.pipeline_utils import (  # Import custom classes/functions
        TemporalFeatureEngineer,
        average_rides_last_4_weeks,
    )

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    models = model_registry.get_models(name=config.MODEL_NAME)
    model = max(models, key=lambda model: model.version)
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / "lgb_model.pkl")

    return model


def load_metrics_from_registry(version=None):

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    models = model_registry.get_models(name=config.MODEL_NAME)
    model = max(models, key=lambda model: model.version)

    return model.training_metrics


def fetch_next_hour_predictions():
    # Get current UTC time and round up to next hour
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    est = pytz.timezone('US/Eastern')
    next_hour_est = next_hour.astimezone(est)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
    df = fg.read()

    # Then filter for next hour in the DataFrame
    df = df[df["start_hour"] == next_hour]

    df["start_hour"] = df["start_hour"].dt.tz_convert('US/Eastern')

    print(f"Current UTC time: {now}")
    print(f"Next hour UTC: {next_hour}; Next Hour Eastern: {next_hour_est}")
    print(f"Found {len(df)} records")
    return df


def fetch_predictions(hours):
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)

    df = fg.filter((fg.start_hour >= current_hour)).read()
    df["start_hour"] = df["start_hour"].dt.tz_convert('US/Eastern')

    return df


def fetch_hourly_rides(hours):
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    query = fg.select_all()
    query = query.filter(fg.start_hour >= current_hour).read()
    query["start_hour"] = query["start_hour"].dt.tz_convert('US/Eastern')

    return query


def fetch_days_data(days):
    current_date = pd.to_datetime(datetime.now(timezone.utc))

    fetch_data_from = current_date - timedelta(days=(365 + days))
    fetch_data_to = current_date - timedelta(days=365)
    print(fetch_data_from, fetch_data_to)
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    query = fg.select_all()
    # query = query.filter((fg.pickup_hour >= fetch_data_from))
    df = query.read()
    cond = (df["start_hour"] >= fetch_data_from) & (df["start_hour"] <= fetch_data_to)
    df = df[cond]
    df["start_hour"] = df["start_hour"].dt.tz_convert('US/Eastern')

    return df
