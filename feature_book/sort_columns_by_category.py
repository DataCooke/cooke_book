import pandas as pd
import logging


def sort_columns_by_category(df, categories):
    """
    Sorts columns in a DataFrame into categories and returns a dictionary with separate DataFrames.

    Parameters:
    df (pandas.DataFrame): The DataFrame to sort columns.
    categories (dict): A dictionary with column names as keys and category codes as values.

    Returns:
    category_dfs (dict): A dictionary with category codes as keys and DataFrames as values.
    """
    # Create empty dictionaries for each category
    category_columns = {
        't': [],
        'b': [],
        'o': [],
        'n': [],
        'c': [],
        'd': [],
    }

    # Input validation
    if not isinstance(categories, dict):
        raise TypeError("categories must be a dictionary")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Sort column names into categories
    for column_name, category_code in categories.items():
        if category_code in category_columns:
            category_columns[category_code].append(column_name)
        else:
            logging.warning("Invalid category code '%s' for column '%s'", category_code, column_name)

    # Create dataframes for each category using column names
    category_dfs = {code: df[cols] for code, cols in category_columns.items() if cols}

    return category_dfs