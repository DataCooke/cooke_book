import pandas as pd


def filter_columns(column_list, dataframes):
    """
    Filter specified columns from one or multiple pandas DataFrame(s).

    Parameters:
    column_list (list): List of column names to keep in the DataFrame(s).
    dataframes (DataFrame or tuple): DataFrame or tuple of DataFrames to filter.

    Returns:
    DataFrame or tuple: A single DataFrame if one was passed, otherwise a tuple of filtered DataFrames.
    """
    if not isinstance(column_list, list):
        raise ValueError("column_list must be a list of column names.")

    # Convert a single dataframe to a tuple with a single element
    if isinstance(dataframes, pd.DataFrame):
        dataframes = (dataframes,)
    elif not isinstance(dataframes, tuple):
        raise ValueError("dataframes must be a DataFrame or a tuple of DataFrames.")

    # Create a new list to store the filtered dataframes
    filtered_dataframes = []

    # Iterate over the input dataframes
    for df in dataframes:
        # Validate dataframe and columns
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Each element in dataframes must be a DataFrame.")
        
        for column in column_list:
            if column not in df.columns:
                raise ValueError(f"Column {column} not found in DataFrame.")
        
        # Create a new dataframe with only the desired columns
        filtered_df = df[column_list]

        # Add the new dataframe to the filtered_dataframes list
        filtered_dataframes.append(filtered_df)

    # If there is only one filtered dataframe, return it as a single dataframe
    if len(filtered_dataframes) == 1:
        return filtered_dataframes[0]

    # Otherwise, return the filtered dataframes as a tuple
    return tuple(filtered_dataframes)