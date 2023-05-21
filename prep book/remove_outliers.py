def remove_outliers(df, columns):
    """
    Removes rows from a DataFrame that have outlier values in the specified columns.

    An outlier is defined as a value that is more than 3 standard deviations away from the mean.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which to remove outliers.
    columns (list of str): The names of the columns to check for outliers.

    Returns:
    df (pandas.DataFrame): The DataFrame with outliers removed.
    """
    try:
        # Calculate z-scores for the specified columns
        z_scores = df[columns].apply(
            lambda x: np.abs((x - np.nanmean(x)) / np.nanstd(x))
        )
    except KeyError as e:
        logging.error(f"One or more of the specified columns do not exist in the DataFrame: {e}")
        raise
    except Exception as e:
        logging.exception("An error occurred while calculating z-scores.")
        raise
    
    try:
        # Create a mask for values with z-scores less than 3 or NaN
        filter_mask = np.logical_or(np.isnan(z_scores), z_scores < 3)
    except Exception as e:
        logging.exception("An error occurred while creating the filter mask.")
        raise

    try:
        # Apply the mask to the DataFrame
        df = df[filter_mask.all(axis=1)]
    except Exception as e:
        logging.exception("An error occurred while applying the filter mask.")
        raise
    
    return df
