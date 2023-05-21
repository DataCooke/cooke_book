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
