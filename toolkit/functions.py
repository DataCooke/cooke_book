from scipy import stats
import numpy as np
import pandas as pd

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