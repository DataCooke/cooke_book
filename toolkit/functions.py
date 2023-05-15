from scipy import stats
import numpy as np
import pandas as pd
import logging

def remove_outliers(df, columns):
    # Calculate z-scores for the specified columns
    z_scores = df[columns].apply(
        lambda x: np.abs((x - np.nanmean(x)) / np.nanstd(x))
    )
    
    # Create a mask for values with z-scores less than 3 or NaN
    filter_mask = np.logical_or(np.isnan(z_scores), z_scores < 3)
    
    # Apply the mask to the DataFrame
    df = df[filter_mask.all(axis=1)]
    
    return df



def validate_dict(dict_keys, df):
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


def sort_columns_by_category(df, categories):
    # Create empty dictionaries for each category
    category_columns = {
        't': [],
        'b': [],
        'o': [],
        'n': [],
        'c': [],
        'd': [],
    }

    # Sort column names into categories
    for column_name, category_code in categories.items():
        if category_code in category_columns:
            category_columns[category_code].append(column_name)

    # Create dataframes for each category using column names
    category_dfs = {code: df[cols] for code, cols in category_columns.items() if cols}

    return category_dfs

def date_processing(df):
    new_columns = {}
    us_holidays = holidays.US()
    today = pd.Timestamp.now().floor('D')

    for col in df.columns:
        try:
            datetime_col = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
        except ValueError:
            continue

        # Calculate new features
        days_of_week = datetime_col.dt.dayofweek
        months = datetime_col.dt.month
        years = datetime_col.dt.year
        is_us_holiday = datetime_col.dt.date.isin(us_holidays)
        day_before_holiday = (datetime_col - pd.Timedelta(days=1)).dt.date.isin(us_holidays)
        day_after_holiday = (datetime_col + pd.Timedelta(days=1)).dt.date.isin(us_holidays)
        days_since_today = (today - datetime_col).dt.days

        # Replace NaN values with -1
        for feature in [days_of_week, months, years, days_since_today]:
            feature.fillna(-1, inplace=True)

        # Convert boolean features to int
        for feature in [is_us_holiday, day_before_holiday, day_after_holiday]:
            feature = feature.astype(int)

        # Store new features in the dictionary
        new_columns[col + '_dow'] = days_of_week
        new_columns[col + '_month'] = months
        new_columns[col + '_year'] = years
        new_columns[col + 'us_holiday'] = is_us_holiday
        new_columns[col + '_day_prior_us_holiday'] = day_before_holiday
        new_columns[col + '_day_after_us_holiday'] = day_after_holiday
        new_columns[col + '_days_since_today'] = days_since_today

        df[col] = datetime_col

    new_df = pd.DataFrame(new_columns)
    new_df = new_df.replace(-1, np.nan)

    return new_df


def reduce_cardinality(df, threshold=0.05):
    new_df = df.copy()
    for col in new_df.select_dtypes(include='object'):
        counts = new_df[col].value_counts(normalize=True)
        categories_to_replace = counts[counts <= threshold].index.tolist()
        if len(categories_to_replace) > 3:
            new_df[col] = new_df[col].where(new_df[col].isin(categories_to_replace), 'other').replace(['Other', 'OTHER'], 'other')
        else:
            new_df[col] = new_df[col].replace(['Other', 'OTHER'], 'other')
    return new_df


def binary_encode_categorical_cols(df, cat_cols=None):
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=['object', 'category', 'boolean']).columns
    encoder = ce.BinaryEncoder(cols=cat_cols, handle_missing='return_nan')
    df_encoded = encoder.fit_transform(df)
    
    return df_encoded

