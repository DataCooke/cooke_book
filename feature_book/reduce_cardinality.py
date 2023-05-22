import pandas as pd
import logging


def reduce_cardinality(df, threshold=0.05):
    """
    Reduce the cardinality of categorical features in the DataFrame.
    
    This function merges rare categories (those that constitute less than 'threshold' proportion of the data) into an 'other' category.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    threshold (float): The threshold for determining when to merge categories. Defaults to 0.05.

    Returns:
    new_df (pandas.DataFrame): The DataFrame with reduced cardinality.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a number")
    if not (0 < threshold < 1):
        raise ValueError("threshold must be between 0 and 1")

    new_df = df.copy()
    for col in new_df.select_dtypes(include='object'):
        counts = new_df[col].value_counts(normalize=True)
        categories_to_replace = counts[counts <= threshold].index.tolist()
        if len(categories_to_replace) > 3:
            new_df[col] = new_df[col].where(new_df[col].isin(categories_to_replace), 'other').replace(['Other', 'OTHER'], 'other')
        else:
            new_df[col] = new_df[col].replace(['Other', 'OTHER'], 'other')

    logging.info("Cardinality reduction completed.")
    return new_df