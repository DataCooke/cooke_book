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
