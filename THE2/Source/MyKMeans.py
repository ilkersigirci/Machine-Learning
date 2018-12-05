#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

EPSILON = 0.0001


class MyKMeans:
    """K-Means clustering similar to sklearn 
    library but different.
    https://goo.gl/bnuM33

    But still same.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init_method : string, optional, default: 'random'
        Initialization method. Values can be 'random', 'kmeans++'
        or 'manual'. If 'manual' then cluster_centers need to be set.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    cluster_centers : np.array, used only if init_method is 'manual'.
        If init_method is 'manual' without fitting, these values can be used
        for prediction.
    """

    def __init__(self, init_method="random", n_clusters=3, max_iter=300, random_state=None, cluster_centers=[]):
        self.init_method = init_method
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)
        if init_method == "manual":
            self.cluster_centers = cluster_centers
        else:
            self.cluster_centers = []

    def fit(self, X):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.
        Returns
        ----------
        self : MyKMeans
        """
        pass

    def initialize(self, X):
        """ Initialize centroids according to self.init_method
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.
        Returns
        ----------
        self.cluster_centers : array-like, shape=(n_clusters, n_features)
        """
        pass

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        pass

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        pass


if __name__ == "__main__":
    if __name__ == "__main__":
        X = np.array([[1, 2], [1, 4], [1, 0],
                      [4, 2], [4, 4], [4, 0]])
        kmeans = MyKMeans(n_clusters=2, random_state=0, init_method='kmeans++')
        print kmeans.initialize(X)
        # [[4. 4.]
        #  [1. 0.]]
        kmeans = MyKMeans(n_clusters=2, random_state=0, init_method = 'random')
        print kmeans.initialize(X)
        # [[4. 0.]
        #  [1. 0.]]
        kmeans.fit(X)
        print kmeans.labels
        # array([1, 1, 1, 0, 0, 0])
        print kmeans.predict([[0, 0], [4, 4]])
        # array([1, 0])
        print kmeans.cluster_centers
        # array([[4, 2],
        #       [1, 2]])