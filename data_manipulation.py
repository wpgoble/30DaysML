import numpy as np
import math

def shuffle_data(X, y, seed = None):
    """ Random shuffle on the data in X, y """
    if seed:
        np.random.seed(seed)
    
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    return X[index], y[index]

def train_test_split(X, y, test_size = 0.5, shuffle = True, seed = None):
    """ Splits the data into train and test sets """

    if shuffle:
        X, y = shuffle_data(X, y, seed)
    
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


