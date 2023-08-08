import matplotlib.pyplot as plt
import logging

def plot_variable_importance(df, metric_name=None):
    """
    Plot the performance of the model at each step of the feature elimination.

    Parameters:
    df (pd.DataFrame): DataFrame containing the metric scores, number of features, and weakest feature at each step
    metric_name (str, default=None): Name of the metric used for model evaluation. Used for y-axis label in the plot.

    Returns:
    None
    """
    try:
        # If metric_name is not provided, infer it from the first column of the DataFrame
        if metric_name is None:
            metric_name = df.columns[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_df = df.sort_values(by='num_cols', ascending=False)
        sorted_df['count'] = range(1, len(sorted_df) + 1)
        ax.plot(sorted_df['count'], sorted_df[metric_name], 'o-', markersize=8)
        ax.set_xticks(sorted_df['count'])
        ax.set_xticklabels(sorted_df.index[::-1], rotation=90)
        ax.set_xlabel('Features')
        ax.set_ylabel(metric_name)
        ax.set_title('Feature Selection')
        
        # Determine whether to highlight max or min based on metric_name
        if any(word in metric_name.lower() for word in ['error', 'loss']):
            ax.axhline(sorted_df[metric_name].min(), color='gray', linestyle='--', linewidth=1.5)
        elif any(word in metric_name.lower() for word in ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']):
            ax.axhline(sorted_df[metric_name].max(), color='gray', linestyle='--', linewidth=1.5)
        else:
            logging.warning("Unable to determine whether higher or lower metric values are better. No reference line added.")
            
        ax.invert_xaxis()  # flip the x-axis
        plt.show()
    except Exception as e:
        logging.error("Failed to plot variable importance: %s", str(e))
