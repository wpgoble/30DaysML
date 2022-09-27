from statistics import LinearRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression

from data_manipulation import train_test_split

class Regression(object):
    """
    Creates a base regression model.
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

def main():
    # generate the data and create train and test splits
    X, y = make_regression(n_samples = 100, n_features = 1, noise = 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
    n_samples, n_features = np.shape(X)

    

if __name__ == "__main__":
    main()