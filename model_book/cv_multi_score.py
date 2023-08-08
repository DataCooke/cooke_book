import pandas as pd
import logging
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold

def cv_multi_score(model, predictors, target, scoring=None, folds=5, task='regression'):
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
    task (str): The type of machine learning task - 'classification' or 'regression'.

    Returns:
    scores (dict): A dictionary of the mean and standard deviation of the 
        test scores for each scorer.

    """
    # Set default scoring metrics based on the 'task' parameter
    if scoring is None:
        if task == 'classification':
            scoring = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
        else:
            scoring = ['neg_mean_absolute_error', 'neg_median_absolute_error', 'r2']

    # If it's a classification task, use StratifiedKFold
    if task == 'classification':
        cv = StratifiedKFold(n_splits=folds)
    else:
        cv = KFold(n_splits=folds)

    try:
        cv_results = cross_validate(model, predictors, target, scoring=scoring, cv=cv)
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