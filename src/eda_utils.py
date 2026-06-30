import matplotlib.pyplot as plt
import seaborn as sns

def plot_univariate(df, column, title):
    """Plot a univariate distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(title)
    plt.show()

def plot_bivariate(df, x_col, y_col, title):
    """Plot bivariate relationship."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(title)
    plt.show()
