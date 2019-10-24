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


    def k_neighbour_values(self, xi):
        """
        Determines the values of the k nearest neighbours to a passed entry xi
        """
        distances = np.empty(len(self.X_train))

        for j, x_t in enumerate(self.X_train):
            coord_distance = np.linalg.norm(xi-x_t)
            distances[j] = coord_distance

        k_indices = np.argsort(distances)[:self.k_neighbours]
        k_labels = self.y_train[k_indices]     # labels of the k nearest neighnours, flatten for bincount
        return k_labels


    def predict_clf(self, X):
        """
        predicts the classification labels of the passed matrix
        """
        predictions = np.empty(len(X))

        for i, xi in enumerate(X):  # calculates the distances between the passed coordinate xi and each training point x_t
            k_labels = knn.k_neighbour_values(self, xi)
            class_votes = np.bincount(k_labels)
            split_vote = np.where(class_votes == class_votes.max())[0]  # labels of all votes if equal number of votes between classes
            predictions[i] = np.random.choice(split_vote)  # random choice if more than one prediction

        return predictions


    def predict_reg(self, X):
        """
        UNTESTED!!!!!!!
        predicts the regression values of the passed matrix
        """
        predictions = np.empty(len(X))

        for i, xi in enumerate(X):  # calculates the distances between the passed coordinate xi and each training point x_t
            k_labels = knn.k_neighbour_values(self, xi)
            predictions[i] = k_labels.mean()  # average returned label values

        return predictions
