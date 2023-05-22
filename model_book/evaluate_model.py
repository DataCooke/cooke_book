import pandas as pd
import numpy as np


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model based on the given test data.
    
    Parameters:
    model (sklearn model or xgboost model): Trained model to be evaluated.
    X_test (DataFrame or array-like): Test set features.
    y_test (Series or array-like): Test set target variable.
    
    Returns:
    abs_mean_error (float): Absolute mean error of the model.
    median_abs_error (float): Median absolute error of the model.
    """
    if not hasattr(model, 'predict'):
        raise ValueError("The model should have a 'predict' method.")

    if not isinstance(X_test, (pd.DataFrame, np.ndarray)) or not isinstance(y_test, (pd.Series, np.ndarray)):
        raise ValueError("X_test should be a DataFrame or array-like, and y_test should be a Series or array-like.")
    
    # Predict the target values using the trained model
    y_pred = model.predict(X_test)

    # Calculate the absolute errors
    abs_errors = np.abs(np.ravel(y_test) - y_pred)

    # Calculate the absolute mean error and median absolute error
    abs_mean_error = abs_errors.mean()
    median_abs_error = np.median(abs_errors)

    return abs_mean_error, median_abs_error