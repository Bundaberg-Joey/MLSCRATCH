import numpy as np
import matplotlib.pyplot as plt



class knn(object):
    """
    knn algorithm for classification
    """

    def __init__(self, k=2):
        self.k_neighbours = k
        self.X_train = None
        self.y_train = None


    def fit(self, X, y):
        """ Takes passed training data and updates the knn parameters for accessing later on"""
        self.X_train = np.array(X)
        self.y_train = np.array(y).flatten()


    def predict_clf(self, X):
        """
        predicts the classification of the passed array which are returned as a flattened numpy array
        """
        predictions = np.empty(len(X))

        for i, xi in enumerate(X):  # calculates the distances between the passed coordinate xi and each training point x_t

            distances = np.empty(len(self.X_train))

            for j, x_t in enumerate(self.X_train):
                coord_distance = np.linalg.norm(xi-x_t)
                distances[j] = coord_distance

            k_indices = np.argsort(distances)[:self.k_neighbours]
            k_labels = self.y_train[k_indices].flatten()        # labels of the k nearest neighnours, flatten for bincount
            majority_clf = np.bincount(k_labels).argmax()  # ISSUE : will always classify as the class with lowest magnitude if a 50:50 vote is reached, need to update with an n sided coin flip
            predictions[i] = majority_clf

        return predictions
