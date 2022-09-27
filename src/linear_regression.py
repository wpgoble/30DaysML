import math
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

    def initialize_weights(self, n_features):
        """ Initialize weights randomly [-1/N, 1/N] """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))
    
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # implements gradient descent
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            # calculate L2 loss
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            # gradient of L2 w.r.t weights
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            self.w -= self.learning_rate * grad_w
    
    def predict(self, X):
        X = np.inserrt(X, 0, 1, axis = 1)
        y_pred = X.dot(self.w)
        return y_pred


def main():
    # generate the data and create train and test splits
    X, y = make_regression(n_samples = 100, n_features = 1, noise = 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
    n_samples, n_features = np.shape(X)

    

if __name__ == "__main__":
    main()