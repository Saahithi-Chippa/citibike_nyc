import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel


# Custom transformer for feature selection based on feature importances
class FeatureImportanceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model, top_n=10):
        """
        Selects the top N features based on the feature importances from the provided model.
        
        Parameters:
        ----------
        model : sklearn model
            The trained model used to obtain feature importances.
        top_n : int, optional
            The number of top features to select based on feature importances (default is 10).
        """
        self.model = model
        self.top_n = top_n
        self.selected_features = []

    def fit(self, X, y=None):
        # Fit the model to get feature importances
        self.model.fit(X, y)
        # Get feature importances and sort them in descending order
        importances = self.model.feature_importances_
        # Get the indices of the top N features
        indices = np.argsort(importances)[::-1][:self.top_n]
        self.selected_features = indices
        return self

    def transform(self, X):
        # Return only the selected top N features
        return X.iloc[:, self.selected_features]
    
# Instantiate 
# add_feature_importances = FeatureImportanceSelector()

# Function to calculate the average rides over the last 4 weeks
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    last_4_weeks_columns = [
        f"rides_t-{7*24}",  # 1 week ago
        f"rides_t-{14*24}",  # 2 weeks ago
        f"rides_t-{21*24}",  # 3 weeks ago
        f"rides_t-{28*24}",  # 4 weeks ago
    ]

    # Ensure the required columns exist in the DataFrame
    for col in last_4_weeks_columns:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate the average of the last 4 weeks
    X["average_rides_last_4_weeks"] = X[last_4_weeks_columns].mean(axis=1)

    return X


# FunctionTransformer to add the average rides feature
add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)


# Custom transformer to add temporal features
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["hour"] = X_["start_hour"].dt.hour
        X_["day_of_week"] = X_["start_hour"].dt.dayofweek

        return X_.drop(columns=["start_hour", "start_station_id"])


# Instantiate the temporal feature engineer
add_temporal_features = TemporalFeatureEngineer()

# Custom transformer for PCA feature reduction
class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)
    





# Function to return the pipeline
def get_pipeline(**hyper_params):
    """
    Returns a pipeline with optional parameters for LGBMRegressor.

    Parameters:
    ----------
    **hyper_params : dict
        Optional parameters to pass to the LGBMRegressor.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline with feature engineering and LGBMRegressor.
    """
    pipeline = make_pipeline(
        add_temporal_features,
        lgb.LGBMRegressor(**hyper_params)  # Pass optional parameters here
    )
    return pipeline

# def pca_feature_reduction(X_train, y_train, X_test, top_n=10):
#     """
#     Reduces features using Principal Component Analysis (PCA).
    
#     Parameters:
#         X_train (pd.DataFrame): Training feature set.
#         y_train (pd.Series): Training target set.
#         X_test (pd.DataFrame): Test feature set.
#         top_n (int): The number of top components to keep (default is 10).
    
#     Returns:
#         X_train_reduced (pd.DataFrame): Reduced training features.
#         X_test_reduced (pd.DataFrame): Reduced test features.
#         pca (PCA): Fitted PCA model.
#     """
#     # Apply PCA to reduce features
#     pca = PCA(n_components=top_n)
#     X_train_reduced = pca.fit_transform(X_train)
#     X_test_reduced = pca.transform(X_test)
    
#     return X_train_reduced, X_test_reduced, pca

# def train_lgbm_with_reduced_features(X_train_reduced, y_train, X_test_reduced, y_test):
#     """
#     Trains a LightGBM model on the reduced feature set and evaluates it.
    
#     Parameters:
#         X_train_reduced (pd.DataFrame or np.array): Reduced training feature set.
#         y_train (pd.Series): Training target set.
#         X_test_reduced (pd.DataFrame or np.array): Reduced test feature set.
#         y_test (pd.Series): Test target set.
    
#     Returns:
#         model (lgb.Booster): Trained LightGBM model.
#         predictions (np.array): Predictions from the model for the test set.
#         mae (float): Mean Absolute Error of the model's predictions.
#     """
#     # Train the LightGBM model
#     model = lgb.LGBMRegressor(objective='regression', metric='mae')
#     model.fit(X_train_reduced, y_train)
    
#     # Predict on the test data
#     predictions = model.predict(X_test_reduced)
    
#     # Calculate MAE (Mean Absolute Error)
#     # mae = mean_absolute_error(y_test, predictions)
    
#     return model, predictions

# Function to return the pipeline with PCA and temporal feature engineering
def get_pca_pipeline(**hyper_params):
    """
    Returns a pipeline with temporal feature engineering, PCA, and LGBMRegressor.

    Parameters:
    ----------
    **hyper_params : dict
        Optional parameters to pass to the LGBMRegressor.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline with temporal feature engineering, PCA, and LGBMRegressor.
    """
    pipeline = make_pipeline(
        TemporalFeatureEngineer(),         # Add temporal features
        PCATransformer(n_components=10),   # Apply PCA for dimensionality reduction
        lgb.LGBMRegressor(**hyper_params)  # Pass optional parameters to the LGBMRegressor
    )
    return pipeline




# Function to return the pipeline with feature importance-based selection and LGBMRegressor
def get_pipeline_feature_importance(**hyper_params):
    """
    Returns a pipeline with temporal feature engineering, feature importance-based feature selection, and LGBMRegressor.

    Parameters:
    ----------
    **hyper_params : dict
        Optional parameters to pass to the LGBMRegressor.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline with temporal feature engineering, feature selection, and LGBMRegressor.
    """
    # Initialize the LGBMRegressor
    lgb_model = lgb.LGBMRegressor(**hyper_params)
    
    pipeline = make_pipeline(
        TemporalFeatureEngineer(),             # Add temporal features
        FeatureImportanceSelector(model=lgb_model, top_n=10),  # Apply feature importance-based selection
        lgb_model                             # Train the LGBM model on the reduced feature set
    )
    return pipeline
