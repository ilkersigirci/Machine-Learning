#!/usr/bin/env python
# -*- coding: utf-8 -*-

class MyKNeighborsClassifier:
    """Classifier implementing the k-nearest neighbors vote similar to sklearn 
    library but different.
    https://goo.gl/Cmji3U

    But still same.
    
    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default.
    method : string, optional (default = 'classical')
        method for voting. Possible values:
        - 'classical' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'weighted' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - 'validity' weights are calculated with distance and multiplied
          of validity for each voter.  
        Note: implementing kd_tree is bonus.
    norm : {'l1', 'l2'}, optional (default = 'l2')
        Distance norm. 'l1' is manhattan distance. 'l2' is euclidean distance.
    Examples
    --------
    """
    def __init__(self, n_neighbors=5, method='classical', norm='l2'):
        self.n_neighbors = n_neighbors
        self.method = method
        self.norm = norm
        self.labels = []
        self.distinct_labels = []
        self.data = []

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values
        Parameters
        ----------
        X : array-like, shape (n_query, n_features),
            Training data. 
        y : array-like, shape = [n_samples] 
            Target values.
        """

        pass
            
    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X : array-like, shape (n_query, n_features),
            Test samples.
        Returns
        -------
        y : array of shape [n_samples]
            Class labels for each data sample.
        """
        if len(self.labels) == 0:
            raise ValueError("You should fit first!")
            
        pass
        
    def predict_proba(self, X, method=None):
        """Return probability estimates for the test data X.
        Parameters
        ----------
        X : array-like, shape (n_query, n_features),
            Test samples.
        method : string, if None uses self.method.
        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        pass

if __name__=='__main__':
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh = MyKNeighborsClassifier(n_neighbors=3, method="validity")
    neigh.fit(X, y)
    n = 0.9
    print(neigh.predict_proba([[n]], method='classical'))
    # [[0.66666667 0.33333333]]
    print(neigh.predict_proba([[n]], method='weighted'))
    # [[0.75089392 0.24910608]]
    print(neigh.predict_proba([[n]], method='validity'))
    # [[0.75697674 0.24302326]]