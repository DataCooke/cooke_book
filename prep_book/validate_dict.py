import pandas as pd
import logging


def validate_dict(dict_keys, df):
    """
    Validates and orders columns in a DataFrame based on a dictionary.

    Parameters:
    dict_keys (dict): A dictionary with column names as keys.
    df (pandas.DataFrame): The DataFrame to validate and order.

    Returns:
    validated_df (pandas.DataFrame): A DataFrame with validated and ordered columns.
    """
    # Input validation
    if not isinstance(dict_keys, dict):
        raise TypeError("dict_keys must be a dictionary")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    dict_keys_set = set(dict_keys.keys())
    df_columns_set = set(df.columns)

    missing_keys = list(dict_keys_set - df_columns_set)
    extra_columns = list(df_columns_set - dict_keys_set)
    ordered_cols = [col for col in dict_keys if col in df_columns_set]

    validated_df = df[ordered_cols]

    # Improved logging
    if missing_keys:
        logging.warning("Missing keys: %s", missing_keys)
    if extra_columns:
        logging.warning("Extra columns: %s", extra_columns)
    
    return validated_df