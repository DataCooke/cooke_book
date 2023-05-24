import pandas as pd


def increment_data(dataframe, target_column, increment_value, n_increments, direction='both'):
    """
    Increment a specified target_column in the dataframe and add a new row for each increment.
    
    Parameters:
    dataframe (DataFrame): The input dataframe.
    target_column (str): The column to be incremented.
    increment_value (float): The value by which to increment the predictor.
    n_increments (int): The number of increments.
    direction (str, optional): The direction of increment ('ascending', 'descending', or 'both'). Default is 'both'.
    
    Returns:
    DataFrame: The dataframe with incremented data.
    """
    # Error handling
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The input should be a pandas DataFrame.")
    
    if target_column not in dataframe.columns:
        raise ValueError(f"The target_column should be one of the dataframe's columns. Got {predictor} instead.")
    
    if direction not in ['ascending', 'descending', 'both']:
        raise ValueError("The direction should be one of 'ascending', 'descending', or 'both'.")
    
    # Copy the dataframe
    df = dataframe.copy()
    
    # Calculate the increments
    for i in range(1, n_increments + 1):
        current_value = df[target_column].iloc[-1]
        
        if direction == 'ascending':
            increment = increment_value
        elif direction == 'descending':
            increment = -increment_value
        else: # direction == 'both'
            increment = increment_value * i if i % 2 == 0 else -increment_value * i
        
        new_value = current_value + increment
        new_row = df.iloc[[-1]].copy()
        new_row[target_column] = new_value
        
        # Add the new row to the dataframe
        df = df.append(new_row, ignore_index=True)
    
    # Sort the dataframe
    df.sort_values(by=target_column, inplace=True, ignore_index=True)
    
    return df