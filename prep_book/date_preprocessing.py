import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import logging


def date_preprocessing(df):
    """
    Process date columns in a DataFrame and generate new features.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.

    Returns:
    new_df (pandas.DataFrame): A DataFrame with new date-related features.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    new_columns = {}
    cal = calendar()
    us_holidays = cal.holidays(start='1900-01-01', end='2099-12-31')
    today = pd.Timestamp.now().floor('D')

    for col in df.columns:
        try:
            datetime_col = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
        except Exception as e:
            logging.warning(f"Error processing column '{col}': {str(e)}")
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
        new_columns[col + '_us_holiday'] = is_us_holiday
        new_columns[col + '_day_prior_us_holiday'] = day_before_holiday
        new_columns[col + '_day_after_us_holiday'] = day_after_holiday
        new_columns[col + '_days_since_today'] = days_since_today

        df[col] = datetime_col

    new_df = pd.DataFrame(new_columns)
    new_df = new_df.replace(-1, np.nan)

    return new_df