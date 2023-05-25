import pandas as pd
import logging


def get_top_features(df, num_features):
    """
    Retrieve top features from the feature selection DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the metric scores, number of features, and weakest feature at each step
    num_features (int): Number of top features to retrieve

    Returns:
    top_features (list): List of top feature names
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if not isinstance(num_features, int):
            raise ValueError("num_features must be an integer.")
        if num_features <= 0:
            raise ValueError("num_features must be greater than 0.")

        sorted_df = df.sort_index(ascending=False)
        sorted_df = sorted_df['weakest feature']
        top_features = sorted_df.head(num_features).tolist()
        return top_features
    except Exception as e:
        logging.error("Failed to get top features: %s", str(e))
        return None