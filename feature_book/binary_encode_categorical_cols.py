import logging
import pandas as pd
import category_encoders as ce


def binary_encode_categorical_cols(df, cat_cols=None):
    """
    Binary encode categorical columns in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    cat_cols (list): List of categorical column names to encode. If not provided, it will default to all 
                      object, category, and boolean datatype columns.

    Returns:
    df_encoded (pandas.DataFrame): The DataFrame with encoded categorical columns.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if cat_cols is not None:
        if not isinstance(cat_cols, list):
            raise TypeError("cat_cols must be a list")
        if not all(isinstance(col, str) for col in cat_cols):
            raise TypeError("cat_cols must be a list of strings")
        if not set(cat_cols).issubset(df.columns):
            raise ValueError("All cat_cols must be columns in df")

    # If cat_cols is not provided, default to all object, category, and boolean datatype columns
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=['object', 'category', 'boolean']).columns

    # Perform binary encoding
    try:
        encoder = ce.BinaryEncoder(cols=cat_cols, handle_missing='return_nan')
        df_encoded = encoder.fit_transform(df)
    except Exception as e:
        logging.error("Error during binary encoding: %s", str(e))
        raise

    logging.info("Binary encoding completed.")
    return df_encoded