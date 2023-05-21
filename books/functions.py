# Standard library imports
import logging
import time

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb



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


def validate_dict(dict_keys, df):
    """
    Validates and orders columns in a DataFrame based on a dictionary.

    Parameters:
    dict_keys (dict): A dictionary with column names as keys.
    df (pandas.DataFrame): The DataFrame to validate and order.

    Returns:
    validated_df (pandas.DataFrame): A DataFrame with validated and ordered columns.
    """
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


def date_processing(df):
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


def calculate_vif_numpy(df, target, vifMax, exclude_cols=None):
    """
    This function calculates the Variance Inflation Factor (VIF) for all the features in a dataset, drops the feature with 
    the highest VIF, and repeats the process until all features have VIF less than a specified threshold.

    Parameters:
    df (pd.DataFrame): The input dataframe
    target (pd.Series): The target variable
    vifMax (float): The maximum acceptable VIF. Any feature with a VIF above this value will be dropped.
    exclude_cols (list): A list of column names to be excluded from VIF calculation.

    Returns:
    list: The remaining features after dropping features with high VIF.
    pd.DataFrame: A dataframe containing VIF calculations for each iteration.
    list: A list of correlation matrices for each iteration.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')
    if not isinstance(target, pd.Series):
        raise TypeError('target must be a pandas Series')

    start_time = time.time()

    vif_data = []
    corr_matrices = []
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Exclude specified columns
    if exclude_cols:
        df = df.drop(columns=exclude_cols)
    
    # Convert data frame to numpy array
    X = df.values
    y = target.values
    
    iteration = 0
    while True:
        # Calculate VIF for all features
        vif = pd.DataFrame()
        vif["variables"] = df.columns
        vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif["iteration"] = iteration
        vif_data.append(vif.copy())
        
        # Calculate correlation matrices for top 2 features at each iteration
        top_vif_indices = vif["VIF"].nlargest(2).index
        selected_features = df.columns[top_vif_indices]
        selected_X = X[:, top_vif_indices]
        selected_X_df = pd.DataFrame(selected_X, columns=selected_features)
        selected_X_df['target'] = y
        corr_matrix_df = selected_X_df.corr()
        corr_matrices.append(corr_matrix_df)
        
        # Check if maximum VIF is below threshold
        if vif["VIF"].max() <= vifMax:
            break

        # Drop the feature with the highest VIF
        max_vif_indices = vif["VIF"].nlargest(2).index
        max_vif_index = max_vif_indices[np.argmin(np.abs(corr_matrix_df.iloc[:-1, -1].values))]
        X = np.delete(X, max_vif_index, axis=1)
        df = df.drop(columns=[df.columns[max_vif_index]])
        
        iteration += 1

    end_time = time.time()
    logging.info(f"Execution time: {round(end_time - start_time, 2)} seconds")

    return list(df.columns), pd.concat(vif_data), corr_matrices


def cv_multi_score(model, predictors, target, scoring, folds):
    """
    Calculate cross-validation scores for a given model.

    Parameters:
    model (sklearn.base.BaseEstimator): The model to evaluate.
    predictors (pandas.DataFrame or numpy.array): The predictor variables.
    target (pandas.Series or numpy.array): The target variable.
    scoring (str, callable, list/tuple, dict or None): A single string scorer 
        callable or a list/tuple of single scorer callables or a dict object 
        with keys being the scorer names and values the scorer callables.
    folds (int or cross-validation generator): The number of folds in K-fold 
        cross validation, or a cross-validation generator.

    Returns:
    scores (dict): A dictionary of the mean and standard deviation of the 
        test scores for each scorer.
    """
    try:
        cv_results = cross_validate(model, predictors, target, scoring=scoring, cv=folds)
    except Exception as e:
        logging.exception("An error occurred during cross validation.")
        raise e
    
    scores = {}
    for metric in scoring:
        try:
            scores[metric] = {
                'mean': cv_results[f'test_{metric}'].mean(),
                'std': cv_results[f'test_{metric}'].std()
            }
        except KeyError:
            logging.error(f"'test_{metric}' not found in cv_results.")
            raise KeyError(f"'test_{metric}' not found in cv_results.")
        except Exception as e:
            logging.exception("An error occurred when calculating mean and standard deviation of scores.")
            raise e

    return scores


def feature_elimination(X, y, model, metric=mean_absolute_error, n_jobs=1):
    """
    Perform backward feature elimination on a given model and dataset.

    Parameters:
    X (pd.DataFrame): Independent features
    y (pd.Series): Target variable
    model (sklearn estimator): The model to use for feature importance
    metric (callable, default=mean_absolute_error): Metric to use for model evaluation. Should be a callable
        that takes two arguments (the true values and the predicted values, in that order) and returns a scalar value where lower values are better.
    n_jobs (int, default=1): Number of cores to run in parallel while fitting across folds. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

    Returns:
    pd.DataFrame: DataFrame containing the metric score, number of features, and weakest feature at each step
    """
    start_time = time.time()
    num_cols = X.shape[1]
    metric_scores = []
    num_cols_list = []
    eliminated_features = []
    
    for i in range(num_cols):
        model.fit(X, y)
        y_pred = model.predict(X)
        score = metric(y, y_pred)
        metric_scores.append(score)
        num_cols_list.append(num_cols)

        # find the weakest feature
        idx = np.argmin(model.feature_importances_)
        weakest_feat = X.columns[idx]
        
        # remove the weakest feature
        X = X.drop(columns=[weakest_feat])
        num_cols = len(X.columns)
        eliminated_features.append(weakest_feat)
        
        if num_cols == 1:
            remaining_col = X.columns[-1]
            break

    feature_elim_df = pd.DataFrame({
        'metric_score': metric_scores,
        'num_cols': num_cols_list,
        'eliminated_feature': eliminated_features
    })

    feature_elim_df.loc[len(feature_elim_df)] = [None, None, remaining_col]

    end_time = time.time()
    logging.info(f"Execution time: {round(end_time - start_time, 2)} seconds")

    return feature_elim_df


def plot_variable_importance(df, metric_name='Metric Score'):
    """
    Plot the performance of the model at each step of the feature elimination.

    Parameters:
    df (pd.DataFrame): DataFrame containing the metric scores, number of features, and weakest feature at each step
    metric_name (str, default='Metric Score'): Name of the metric used for model evaluation. Used for y-axis label in the plot.

    Returns:
    None
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_df = df.sort_values(by='num_cols', ascending=False)
        sorted_df['count'] = range(1, len(sorted_df) + 1)
        ax.plot(sorted_df['count'], sorted_df['metric_score'], 'o-', markersize=8)
        ax.set_xticks(sorted_df['count'])
        ax.set_xticklabels(sorted_df.index[::-1], rotation=90)
        ax.set_xlabel('Features')
        ax.set_ylabel(metric_name)
        ax.set_title('Feature Selection')
        ax.axhline(sorted_df['metric_score'].min(), color='gray', linestyle='--', linewidth=1.5)
        ax.invert_xaxis()  # flip the x-axis
        plt.show()
    except Exception as e:
        logging.error("Failed to plot variable importance: %s", str(e))


def get_top_features(df, num_features):
    """
    Retrieve top features from the feature selection DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the metric scores, number of features, and weakest feature at each step
    num_features (int): Number of top features to retrieve

    Returns:
    top_features (list): List of top feature names
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if not isinstance(num_features, int):
            raise ValueError("num_features must be an integer.")
        if num_features <= 0:
            raise ValueError("num_features must be greater than 0.")

        sorted_df = df.sort_index(ascending=False)
        sorted_df = sorted_df['weakest feature']
        top_features = sorted_df.head(num_features).tolist()
        return top_features
    except Exception as e:
        logging.error("Failed to get top features: %s", str(e))
        return None


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


def xgb_hyperopt(X, y, n_iterations, param_ranges=None, eval_metric='mae', objective='reg:absoluteerror', booster='gbtree', nfold=5, early_stopping_rounds=10):
    """
    Perform hyperparameter tuning for XGBoost model using Bayesian Optimization.
    
    Parameters:
    X (DataFrame): Features for the model.
    y (Series or array-like): Target variable.
    n_iterations (int): Number of iterations for Bayesian Optimization.
    param_ranges (dict, optional): Ranges for parameters to be optimized. Default is a pre-defined dictionary.
    eval_metric (str, optional): Evaluation metric for the model. Default is 'mae'.
    objective (str, optional): Objective function for the model. Default is 'reg:absoluteerror'.
    booster (str, optional): Booster type for the model. Default is 'gbtree'.
    nfold (int, optional): Number of folds for cross validation. Default is 5.
    early_stopping_rounds (int, optional): Number of rounds without improvement to trigger early stopping. Default is 10.
    
    Returns:
    tuned_model (XGBModel): The tuned XGBoost model.
    params (dict): The parameters used for the tuned model.
    """
    # Default parameter ranges
    if param_ranges is None:
        param_ranges = {'num_boost_round': (50, 300),
                        'max_depth': (3, 7),
                        'subsample': (0.7, 0.9),
                        'colsample_bytree': (0.7, 0.9),
                        'gamma': (0, 0.5),
                        'min_child_weight': (1, 6)}

    def xgb_evaluate(**params):
        for param in ['max_depth', 'num_boost_round']:
            params[param] = int(params[param])
        params['eval_metric'] = eval_metric
        params['objective'] = objective
        params['booster'] = booster
        params['nthread'] = -1
        dtrain = xgb.DMatrix(X, label=y)
        cv_result = xgb.cv(params, dtrain, num_boost_round=params['num_boost_round'], nfold=nfold, early_stopping_rounds=early_stopping_rounds)
        return -1.0 * cv_result['test-'+eval_metric+'-mean'].iloc[-1]

    # Error handling
    if not isinstance(X, pd.DataFrame) or not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError("X should be a DataFrame and y should be a Series or array-like.")
    if n_iterations <= 0:
        raise ValueError("n_iterations should be greater than 0.")
    if nfold <= 0 or not isinstance(nfold, int):
        raise ValueError("nfold should be an integer greater than 0.")
    if early_stopping_rounds <= 0 or not isinstance(early_stopping_rounds, int):
        raise ValueError("early_stopping_rounds should be an integer greater than 0.")

    xgb_bo = BayesianOptimization(xgb_evaluate, param_ranges)
    xgb_bo.maximize(init_points=5, n_iter=n_iterations)

    params = xgb_bo.max['params']
    params['max_depth'] = int(params['max_depth'])
    params['num_boost_round'] = int(params['num_boost_round'])
    params['eval_metric'] = eval_metric
    params['objective'] = objective
    params['booster'] = booster

    tuned_model = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=params['num_boost_round'])

    return tuned_model, params


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model based on the given test data.
    
    Parameters:
    model (sklearn model or xgboost model): Trained model to be evaluated.
    X_test (DataFrame or array-like): Test set features.
    y_test (Series or array-like): Test set target variable.
    
    Returns:
    abs_mean_error (float): Absolute mean error of the model.
    median_abs_error (float): Median absolute error of the model.
    """
    if not hasattr(model, 'predict'):
        raise ValueError("The model should have a 'predict' method.")

    if not isinstance(X_test, (pd.DataFrame, np.ndarray)) or not isinstance(y_test, (pd.Series, np.ndarray)):
        raise ValueError("X_test should be a DataFrame or array-like, and y_test should be a Series or array-like.")
    
    # Predict the target values using the trained model
    y_pred = model.predict(X_test)

    # Calculate the absolute errors
    abs_errors = np.abs(np.ravel(y_test) - y_pred)

    # Calculate the absolute mean error and median absolute error
    abs_mean_error = abs_errors.mean()
    median_abs_error = np.median(abs_errors)

    return abs_mean_error, median_abs_error


def increment_data(dataframe, predictor, increment_value, n_increments, direction='both'):
    """
    Increment a specified predictor in the dataframe and add a new row for each increment.
    
    Parameters:
    dataframe (DataFrame): The input dataframe.
    predictor (str): The predictor to be incremented.
    increment_value (float): The value by which to increment the predictor.
    n_increments (int): The number of increments.
    direction (str, optional): The direction of increment ('ascending', 'descending', or 'both'). Default is 'both'.
    
    Returns:
    DataFrame: The dataframe with incremented data.
    """
    # Error handling
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The input should be a pandas DataFrame.")
    
    if predictor not in dataframe.columns:
        raise ValueError(f"The predictor should be one of the dataframe's columns. Got {predictor} instead.")
    
    if direction not in ['ascending', 'descending', 'both']:
        raise ValueError("The direction should be one of 'ascending', 'descending', or 'both'.")
    
    # Copy the dataframe
    df = dataframe.copy()
    
    # Calculate the increments
    for i in range(1, n_increments + 1):
        current_value = df[predictor].iloc[-1]
        
        if direction == 'ascending':
            increment = increment_value
        elif direction == 'descending':
            increment = -increment_value
        else: # direction == 'both'
            increment = increment_value * i if i % 2 == 0 else -increment_value * i
        
        new_value = current_value + increment
        new_row = df.iloc[[-1]].copy()
        new_row[predictor] = new_value
        
        # Add the new row to the dataframe
        df = df.append(new_row, ignore_index=True)
    
    # Sort the dataframe
    df.sort_values(by=predictor, inplace=True, ignore_index=True)
    
    return df


def predict_value(model, input_data, sample=1):
    """
    Predicts values using the model for a sample from the input data.
    
    Parameters:
    model (sklearn model or similar): Trained model to make predictions.
    input_data (DataFrame): Input data for the model.
    sample (int, optional): Number of samples to predict. Default is 1.
    
    Returns:
    DataFrame: DataFrame of input values and corresponding predictions.
    """
    if not hasattr(model, 'predict'):
        raise ValueError("The model should have a 'predict' method.")

    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("The input data should be a pandas DataFrame.")
    
    if sample <= 0:
        raise ValueError("The sample size should be a positive integer.")
    
    # Randomly sample data from the input
    sample_data = input_data.sample(n=sample)
    
    # Get the input headers and values
    input_headers = list(sample_data.columns)
    input_values = np.atleast_2d(sample_data.values)
    
    # Make predictions using the model
    predictions = model.predict(input_values)
    
    # Append predictions to input values
    results = np.hstack((input_values, predictions.reshape(-1,1)))
    
    # Append 'target' to input headers
    result_headers = input_headers + ['target']
    
    # Return a DataFrame with results and headers
    return pd.DataFrame(data=results, columns=result_headers)


def draw_lineplot(x_values, y_values, title=None):
    """
    Draw a line plot with the given x and y values.
    
    Parameters:
    x_values (Series or array-like): The x values for the plot.
    y_values (Series or array-like): The y values for the plot.
    title (str, optional): The title of the plot. If not provided, defaults to 'Line plot of Y vs X'.
    
    Returns:
    None
    """
    if not isinstance(x_values, (pd.Series, np.ndarray)) or not isinstance(y_values, (pd.Series, np.ndarray)):
        raise ValueError("x_values and y_values should be Series or array-like.")
    
    if title is None:
        title = f'Line plot of {y_values.name} vs {x_values.name}'

    plt.plot(x_values, y_values)
    plt.xlabel(x_values.name)
    plt.ylabel(y_values.name)
    plt.title(title)
    plt.show()

