"""
K-means clustering is an unsupervised learning algorithm, which groups an 
unlabeled dataset into different clusters. The "K" refers to the number of 
pre-defined clusters the dataset is grouped into.

Plainly, the algorithm entails the following steps:

1. Randomly initialize K cluster centroids i.e. the center of the clusters.
2. Repeat till convergence or end of max number of iterations:
    3. For samples i=1 to m in the dataset:
        Assign the closest cluster centroid to X[i]
    4. For cluster k=1 to K:
        Find new cluster centroids by calculating the mean of the points assigned to cluster k.
"""

import numpy as np

def initialize_random_centroids(K, X):
    """ 
    Initialize and return k random centroids 
    a centroid should be of shape (1, n), so the centroids 
    array will be of shape (K, n)
    """
    m,n = np.shape(X)
    centroids = np.empty((K, n))

    for i in range(K):
        centroids[i] = X[np.random.choice(range(m))] 
    return centroids

def euclidean_distance(x1, x2):
    """
    Calculates and returns the euclidean distance between two vectors x1 and x2
    """
    return np.sqrt(np.sum(np.power(x1 - x2, 2)))

