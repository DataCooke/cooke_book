import pandas as pd
import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization


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