import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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