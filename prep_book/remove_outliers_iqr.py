def remove_outliers_iqr(df, columns):
    """
    Removes rows from a DataFrame that have outlier values in the specified columns.

    An outlier is defined as a value that is less than Q1 - 1.5 * IQR or greater than Q3 + 1.5 * IQR.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which to remove outliers.
    columns (list of str): The names of the columns to check for outliers.

    Returns:
    df (pandas.DataFrame): The DataFrame with outliers removed.
    """
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Only keep rows in the dataframe that have values within the IQR
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

    return df