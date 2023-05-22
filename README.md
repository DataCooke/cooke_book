# Cooke_book

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/downloads/)

## Description

Cooke_book is a collection of Python functions designed to support data science projects. It provides a set of tools and utilities that can simplify various aspects of data generation, feature engineering, statistical inference, modeling, data preparation, hyperparameter tuning, and visualization. The functions in the `cooke_book` package are created to be practical, efficient, and easily integrated into your data science workflows.

## Installation

To install `cooke_book`, you can use pip:

```shell
pip install cooke_book
```


## Folders and Modules

The cooke_book package is organized into the following folders, each containing relevant modules:

* data_generation_book

	* Intent: This folder is dedicated to data generation modules, which are responsible for creating or generating synthetic data for various purposes, such as testing, prototyping, or augmenting existing datasets.

	* Modules: The modules within this folder should include functions or classes that generate synthetic data using different techniques, algorithms, or models. Examples of modules could be data generators for specific distributions, data augmentation techniques, or data synthesis methods.

* feature_book

	* Intent: This folder focuses on feature engineering modules, which are responsible for transforming and manipulating raw data into meaningful features that can be used for model training and analysis.
	
	* Modules: The modules within this folder should include functions or classes that perform feature engineering tasks such as encoding categorical variables, scaling numerical features, creating interaction or polynomial features, or extracting information from text or time series data.

* inference_book
	
	* Intent: This folder is dedicated to inference modules, which involve making predictions or drawing insights from trained models on new or unseen data.
	
	* Modules: The modules within this folder should include functions or classes that encapsulate the logic for model inference, including loading the trained model, preprocessing input data, making predictions, and post-processing the results. These modules can be specific to different types of models, such as regression, classification, time series forecasting, or natural language processing models.

* model_book

	* Intent: This folder focuses on model development and training modules, which are responsible for building, training, and evaluating machine learning or statistical models.
	
	* Modules: The modules within this folder should include functions or classes that define the architecture or structure of the model, handle data preprocessing and splitting, perform model training using appropriate algorithms, tune hyperparameters, and evaluate model performance using relevant metrics or techniques. Modules may cover a wide range of models, such as linear regression, decision trees, support vector machines, or deep neural networks.

* prep_book

	* Intent: This folder is dedicated to data preprocessing modules, which involve cleaning, transforming, and preparing raw data for analysis or modeling tasks.
	
	* Modules: The modules within this folder should include functions or classes that handle tasks such as data cleaning, handling missing values, handling outliers, standardizing or normalizing data, and splitting data into training and testing sets. These modules provide the necessary preprocessing steps to ensure data quality and suitability for downstream analysis or modeling.

* tuning_book

	* Intent: This folder focuses on hyperparameter tuning modules, which involve optimizing the hyperparameters of machine learning models to improve their performance.
	
	* Modules: The modules within this folder should include functions or classes that define hyperparameter search spaces, implement search algorithms such as grid search, random search, or Bayesian optimization, and evaluate different combinations of hyperparameters to find the optimal configuration. These modules help automate the process of hyperparameter tuning and enhance model performance.

* viz_book

	* Intent: This folder is dedicated to visualization modules, which involve creating visual representations of data, models, and results to facilitate understanding and interpretation.

	* Modules: The modules within this folder should include functions or classes that generate various types of visualizations, such as plots, charts, graphs, or interactive visualizations. These modules can cover tasks such as data exploration, feature analysis, model evaluation, result interpretation, or communicating insights from the data and models to stakeholders.

# Module Descriptions (cooke books)

## data_generation_book

### increment_data

Module Description: This module provides a function to increment a specified predictor in a pandas DataFrame and add a new row for each increment.

...
## feature_book

### binary_encode_categorical_cols

Module Description: This module provides a function to perform binary encoding of categorical columns in a pandas DataFrame. Binary encoding is a technique used to transform categorical variables into numerical representation by encoding each category as a binary code. The function takes a DataFrame as input and encodes the specified categorical columns using the category_encoders library. If no categorical columns are provided, it defaults to encoding all object, category, and boolean datatype columns in the DataFrame. The encoded DataFrame is returned as output.

Note: This module requires the category_encoders library to be installed.

### calculate_vif_numpy

Module Description: This module provides a function to calculate the Variance Inflation Factor (VIF) for all the features in a dataset and drop the feature with the highest VIF iteratively until all features have VIF less than a specified threshold. VIF is a measure of multicollinearity, indicating how much the variance of an estimated regression coefficient is increased due to multicollinearity in the dataset. The function takes a pandas DataFrame as input along with the target variable and the maximum acceptable VIF. It also provides an option to exclude specific columns from the VIF calculation. The function returns the remaining features after dropping features with high VIF, a DataFrame containing VIF calculations for each iteration, and a list of correlation matrices for each iteration.

Note: This module requires the pandas, numpy, sklearn.impute, and statsmodels.stats.outliers_influence libraries to be installed.

### reduce_cardinality

Module Description: This module provides a function to reduce the cardinality of categorical features in a DataFrame. Cardinality refers to the number of unique values in a categorical feature. High cardinality can lead to overfitting and model complexity. The function merges rare categories, defined as those constituting less than a specified threshold proportion of the data, into an 'other' category. This helps to reduce the number of categories and improve model robustness. The function takes a pandas DataFrame as input along with an optional threshold value (default is 0.05). It returns a new DataFrame with reduced cardinality.

Note: This module requires the pandas library to be installed.

### sort_columns_by_category

Module Description: This module provides a function to sort columns in a DataFrame into categories based on specified category codes. The function takes a pandas DataFrame and a dictionary of column names and category codes as input. It then sorts the columns into separate DataFrames based on their category codes and returns a dictionary with category codes as keys and corresponding DataFrames as values.

The purpose of this function is to facilitate the organization and analysis of columns based on their categories. Categories can be defined based on specific criteria or classification schemes relevant to the dataset. By sorting the columns into separate DataFrames, it becomes easier to perform targeted operations or analyses on specific categories of columns.

Note: This module requires the pandas library to be installed.

...
## inference_book

### predict_value

Module Description: This module provides a function to make predictions using a trained model on a sample of input data. The function takes a model object, an input DataFrame, and an optional sample size as input. It then randomly samples the input data based on the specified sample size and makes predictions using the model. The function returns a DataFrame containing the input values along with their corresponding predictions.

The purpose of this function is to facilitate the prediction process for machine learning models. By providing the trained model and input data, users can quickly obtain predictions for a sample of data. This can be useful for evaluating model performance, exploring predictions on a subset of data, or generating insights from the model's output.

Note: This module requires the pandas and numpy libraries to be installed.

...
## model_book

### cv_multi_score

Module Description: This module provides a function to calculate cross-validation scores for a given model. The function takes a model object, predictor variables, target variable, scoring metric(s), and the number of folds or a cross-validation generator as input. It performs cross-validation by fitting the model on different subsets of the data and evaluating the performance using the specified scoring metric(s). The function returns a dictionary containing the mean and standard deviation of the test scores for each scorer.

The purpose of this function is to facilitate the evaluation of machine learning models through cross-validation. By providing the model, predictors, target, and scoring metric(s), users can obtain the performance metrics of the model across different folds of the data. This allows for a more robust assessment of the model's performance and helps identify potential issues such as overfitting or underfitting.

Note: This module requires the pandas and logging libraries to be installed, as well as the scikit-learn library for cross-validation functionality.

### evaluate_model

Module Description: This module provides a function to evaluate a trained model based on the given test data. The function takes a model object, test set features (X_test), and test set target variable (y_test) as input. It predicts the target values using the trained model and calculates the absolute mean error and median absolute error between the predicted and actual target values. The function returns the absolute mean error and median absolute error as floating-point values.

The purpose of this function is to assess the performance of a machine learning model on unseen test data. By comparing the predicted values with the actual values, users can obtain insights into the model's accuracy and performance. The absolute mean error provides an average measure of the magnitude of errors, while the median absolute error represents the central tendency of the errors, which is less affected by outliers.

Note: This module requires the pandas and numpy libraries to be installed. The model object should have a 'predict' method to make predictions. The X_test should be a DataFrame or array-like object, and the y_test should be a Series or array-like object.

### feature_elimination

Module Description: The feature_elimination module provides a function to perform backward feature elimination on a given model and dataset. Backward feature elimination is a feature selection technique that iteratively removes the weakest feature from the dataset until a stopping criterion is met. This process helps identify the most important features for a given model.

The function takes the following parameters: X (the independent features), y (the target variable), model (the model to use for feature importance), metric (the evaluation metric to use for model evaluation, defaulting to mean_absolute_error), and n_jobs (the number of cores to use for parallel fitting, defaulting to 1).

The function fits the model on the entire dataset, calculates the metric score, and identifies the weakest feature based on the feature importances provided by the model. It then removes the weakest feature from the dataset and repeats the process until the stopping criterion is met. The function returns a DataFrame containing the metric score, the number of features at each step, and the eliminated feature at each step.

This module can be useful for feature selection and understanding the importance of different features in a predictive model. It provides insights into the impact of removing features on the model's performance and can help in identifying a subset of relevant features for improved model interpretability and efficiency.

### filter_columns

Module Description: The filter_columns module provides a function to filter specified columns from one or multiple pandas DataFrames. It allows you to select and keep only the desired columns, providing flexibility in data manipulation and analysis.

The function takes two parameters: column_list (a list of column names to keep in the DataFrame(s)) and dataframes (either a single DataFrame or a tuple of DataFrames to filter).

The function validates the inputs, ensuring that column_list is a list and dataframes is either a single DataFrame or a tuple of DataFrames. It iterates over the input dataframes, validating that each element is a DataFrame and that the specified columns exist in each DataFrame. It then creates a new DataFrame for each input DataFrame, containing only the desired columns. The filtered DataFrames are stored in a new list, filtered_dataframes.

If there is only one filtered DataFrame, the function returns it as a single DataFrame. Otherwise, if there are multiple filtered DataFrames, the function returns them as a tuple.

This module can be used to select and extract specific columns from DataFrames, allowing for more focused analysis or data transformation. It simplifies the process of working with column subsets, enabling efficient and streamlined data manipulation workflows.

### get_top_features

Module Description: The get_top_features module provides a function to retrieve the top features from a feature selection DataFrame. It allows you to obtain a list of the top feature names based on specified criteria, such as metric scores or feature elimination steps.

The function takes two parameters: df (a pandas DataFrame containing the metric scores, number of features, and weakest feature at each step) and num_features (the number of top features to retrieve).

The function validates the inputs, ensuring that df is a pandas DataFrame and num_features is an integer greater than 0. It sorts the DataFrame in descending order and extracts the column containing the weakest feature. It then retrieves the specified number of top features from the sorted DataFrame and converts them into a list.

If successful, the function returns the list of top feature names. If an error occurs during the execution, the function logs an error message and returns None.

This module can be used to extract the most important or significant features from a feature selection DataFrame, allowing for further analysis or model building with a reduced set of features. It provides flexibility in selecting the top features based on custom criteria or algorithms.

### plot_variable_importance

Module Description: The plot_variable_importance module provides a function to visualize the performance of a model at each step of feature elimination. It creates a plot showing the metric scores (e.g., evaluation scores) for different numbers of features.

The function takes two parameters: df (a pandas DataFrame containing the metric scores, number of features, and weakest feature at each step) and metric_name (a string representing the name of the metric used for model evaluation, with a default value of 'Metric Score').

The function first attempts to create a plot using matplotlib. It sorts the DataFrame in descending order based on the number of features and assigns a count to each step. It then plots the metric scores against the count, using markers to indicate the scores. The x-axis labels are set as the reversed index of the DataFrame, representing the weakest feature at each step. The y-axis label is set to the specified metric_name. The plot includes a horizontal line representing the minimum metric score. Finally, the plot is displayed.

If an error occurs during the execution, the function logs an error message.

This module can be used to visualize the impact of feature elimination on the performance of a model. It helps in understanding how the model's performance changes as features are removed, providing insights into the importance of each feature in the prediction process.

...
## prep_book

### remove_outliers

Module Description: The remove_outliers module provides a function to remove rows from a DataFrame that have outlier values in the specified columns. An outlier is defined as a value that is more than 3 standard deviations away from the mean.

The function takes two parameters: df (a pandas DataFrame from which to remove outliers) and columns (a list of column names to check for outliers).

The function first calculates the z-scores for the specified columns by subtracting the mean and dividing by the standard deviation. Any values that are more than 3 standard deviations away from the mean or NaN are considered outliers.

If any of the specified columns do not exist in the DataFrame, the function logs an error and raises a KeyError. If an error occurs while calculating the z-scores, the function logs an exception.

Next, the function creates a filter mask by applying the logical OR operation to check for NaN values or z-scores less than 3. The filter mask is used to select rows in the DataFrame that do not contain outliers.

If an error occurs while creating the filter mask or applying it to the DataFrame, the function logs an exception.

Finally, the function returns the DataFrame with outliers removed.

This module can be used to preprocess data by removing outliers, which can have a significant impact on the results of statistical analysis or machine learning models. By removing outliers, the data can be more representative of the underlying distribution and improve the accuracy and reliability of subsequent analyses.

### validate_dict

Module Description: The validate_dict module provides a function to validate and order columns in a DataFrame based on a dictionary. It ensures that the columns in the DataFrame match the keys in the dictionary and orders the columns according to the dictionary.

The function takes two parameters: dict_keys (a dictionary with column names as keys) and df (the DataFrame to validate and order).

The function performs input validation to ensure that dict_keys is a dictionary and df is a pandas DataFrame. It raises a TypeError if the input types are not correct.

The function compares the keys in dict_keys with the columns in df to identify any missing keys or extra columns. It creates a new list, ordered_cols, which contains the keys from dict_keys that exist in df in the same order as they appear in dict_keys.

A new DataFrame, validated_df, is created by selecting only the columns from df that are present in ordered_cols.

The function logs a warning if there are missing keys or extra columns, providing the respective lists of missing keys and extra columns. This allows for easy identification of any discrepancies between the dictionary keys and DataFrame columns.

Finally, the function returns the validated_df, which is the DataFrame with validated and ordered columns.

This module can be used to ensure that the columns in a DataFrame match a predefined set of keys and are ordered according to a specified dictionary. It helps in standardizing the column order and ensuring consistency in downstream analysis or processing.

### date_preprocessing

Module Description: The date_processing module provides a function to process date columns in a DataFrame and generate new date-related features. It enhances the original DataFrame by extracting information such as the day of the week, month, year, and other relevant date-related features.

The function takes one parameter: df (the DataFrame to process).

The function performs input validation to ensure that df is a pandas DataFrame. It raises a TypeError if the input type is not correct.

The function initializes a dictionary, new_columns, to store the new features generated from date processing. It also creates a US Federal Holiday calendar and obtains a list of US holidays for the specified date range.

For each column in the DataFrame, the function attempts to convert the values to datetime format using the specified format. Any exceptions encountered during this process are logged as warnings, and the column is skipped.

For the columns successfully converted to datetime, the function calculates new features such as the day of the week, month, year, whether it is a US holiday, whether the day before or after is a US holiday, and the number of days since the current date.

NaN values in the new features are replaced with -1 for consistency. Boolean features are converted to integer values.

The new features are stored in the new_columns dictionary using the original column name as a prefix. For example, the day of the week feature for a column named 'date' would be stored as 'date_dow'.

The original columns in the DataFrame are replaced with their datetime counterparts.

The function creates a new DataFrame, new_df, using the new_columns dictionary. Any -1 values in new_df are replaced with NaN for improved data representation.

Finally, the function returns the new_df, which is the DataFrame with the new date-related features.

This module can be used to extract and generate additional features from date columns in a DataFrame. It is particularly useful for time series analysis, trend analysis, and building predictive models that involve time-dependent data.

...
## tuning_book

### xgb_hyperopt

Module Description: The xgb_hyperopt module provides a function for performing hyperparameter tuning for an XGBoost model using Bayesian Optimization. It utilizes the BayesianOptimization library to search for the optimal combination of hyperparameters.

The function takes several parameters:

X (DataFrame): The features for the model.
y (Series or array-like): The target variable.
n_iterations (int): The number of iterations for Bayesian Optimization.
param_ranges (dict, optional): Ranges for the parameters to be optimized. It has a default value of a pre-defined dictionary.
eval_metric (str, optional): The evaluation metric for the model. It has a default value of 'mae' (mean absolute error).
objective (str, optional): The objective function for the model. It has a default value of 'reg:absoluteerror' (regression with absolute error).
booster (str, optional): The booster type for the model. It has a default value of 'gbtree'.
nfold (int, optional): The number of folds for cross-validation. It has a default value of 5.
early_stopping_rounds (int, optional): The number of rounds without improvement to trigger early stopping. It has a default value of 10.
The function begins by defining default parameter ranges if they are not provided. These parameter ranges define the search space for the hyperparameters to be tuned.

Next, it defines an inner function xgb_evaluate that takes the hyperparameters as input and evaluates the XGBoost model using cross-validation. It converts certain hyperparameters to integers and sets additional parameters such as the evaluation metric, objective function, booster type, and the number of threads. It then trains the XGBoost model using the specified hyperparameters and performs cross-validation using the specified number of folds and early stopping rounds. The negative mean evaluation metric value is returned as the objective to be minimized.

The function performs input validation to ensure the correct types and values for the parameters. If the inputs are invalid, it raises a ValueError with an appropriate error message.

The function creates an instance of BayesianOptimization and initializes it with the xgb_evaluate function and the parameter ranges. It then maximizes the BayesianOptimization instance by iteratively searching for the optimal hyperparameters. The initial points for the search are set to 5, and the specified number of iterations is performed.

After the optimization, the best set of hyperparameters is obtained from the BayesianOptimization instance. The integer values for the 'max_depth' and 'num_boost_round' parameters are converted to integers. The other necessary parameters such as the evaluation metric, objective function, and booster type are also set.

Finally, the function trains an XGBoost model with the tuned hyperparameters using the entire dataset (X and y). The tuned model and the tuned hyperparameters are returned as the output.

This module provides a convenient way to perform hyperparameter tuning for an XGBoost model using Bayesian Optimization. It helps to automatically search for the optimal combination of hyperparameters, improving the model's performance and generalization capabilities.

...
## viz_book

### draw_lineplot

Module Description: The draw_lineplot module provides a function for drawing a line plot with given x and y values using the Matplotlib library.

The function takes the following parameters:

x_values (Series or array-like): The x values for the plot.
y_values (Series or array-like): The y values for the plot.
title (str, optional): The title of the plot. If not provided, it defaults to 'Line plot of Y vs X'.
The function performs input validation to ensure that x_values and y_values are of type Series or array-like. If they are not, a ValueError is raised with an appropriate error message.

The function then plots the line plot using the plot function from Matplotlib. The x_values are plotted on the x-axis, and the y_values are plotted on the y-axis. The title of the plot is set to the provided title or the default title if not provided. The x-axis label is set to the name of the x_values series, and the y-axis label is set to the name of the y_values series. Finally, the plot is displayed using the show function from Matplotlib.

The function provides a simple way to visualize the relationship between two variables using a line plot. It is useful for understanding trends and patterns in data and can be used for exploratory data analysis or presenting results.


## Documentation

Documentation for `cooke_book` has not been created yet

## License

`cooke_book` is licensed under MIT License. See the LICENSE file for more details.