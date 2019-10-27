import numpy as np


class knn(object):
    """
    knn algorithm capable of classification and regression built from numpy
    """

    def __init__(self, k=5):
        """
        Initialise the knn model with the number of nearest neighbours. Default is set to 5.
        
        :param k: int, number of nearest neighbours to consider when predicting new label values
        """
        self.k_neighbours = k
        self.X_train = None
        self.y_train = None
    

    def fit(self, X, y):
        """ 
        Takes passed training data and updates the knn parameters for accessing later on during prediction.
        knn cannot be explicitly trained in the standard sense, this just updates the model internal variables.
        
        :param X: n x m feature matrix
        :param y: n X 1 score column
        
        :return: None
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y).flatten()
        self.fitted = True
        return None

    
    def k_neighbour_values(self, xi):
        """
        Determines the values of the k nearest neighbours to a passed entry xi using the eucledian distances.
        
        :param xi: np.array, 1 x m feature row vector
        
        :return: k_labels: np.array, flat array containing the values of the nearest k neighbours to the passed entry xi
        """
        distances = np.linalg.norm(xi - self.X_train, axis=1)  # calculate eucledian distances
        k_indices = np.argsort(distances)[:self.k_neighbours]
        k_labels = self.y_train[k_indices]     
        return k_labels


    def predict_clf(self, X):
        """
        Predicts the classification labels of the passed feature matrix by majority vote of the nearest neighbours, random choice if tie.
        
        :param X: np.array, a feature matrix to determine the nearest neighbours of based on passed training data
        
        :return: predictions: np.array, flat array containing majority classifications based on k nearest neighbours.
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
        Predicts the regression values of the passed feature matrix by averaging the values of the nearest neighbours
        
        :param X: np.array, a feature matrix to determine the nearest neighbours of based on passed training data
        
        :return: predictions: np.array, flat array containing averaged regressions based on k nearest neighbours.
        """
        predictions = np.empty(len(X))

        for i, xi in enumerate(X):  # calculates the distances between the passed coordinate xi and each training point x_t
            k_labels = knn.k_neighbour_values(self, xi)
            predictions[i] = k_labels.mean()  # average returned label values

        return predictions

   