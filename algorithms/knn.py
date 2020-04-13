import numpy as np


class Knn(object):
    """Concise implementation of K-nearest algorithms algorithm in numpy for both classification and regression.

    Attributes
    ----------
    k_neighbours : int, (default = 5)
        Number of algorithms to consider during prediction.

    distance : func(xi, X) -->  np.array(), (default = None)
        User supplied distance function with args (xi, X) and returns a flat array of distances.
        If not passed (default) then the internal euclidean distance function is used.

    _X_train : feature matrix for training, shape (num_entries, num_features)
        Training feature data stored here on calling `fit`.

    _y_train : target array for training, shape (num_entries, )
        Target feature data stored here on calling `fit`.

    voter : func(neighbour_data) --> prediction, (default = None)
        User supplied function on how to hande the prediction from algorithms data.
        If not passed (default) then the internal function is used (Mean for regression, majority for classification).

    Notes
    -----
    To match the standard workflow of `fit` then `predict`, Knn does have a `train` method.
    No parameters are learned by the algorithm during this fit but the training data is stored withi the object.
    """

    def __init__(self, k_neighbours=5, distance=None):
        """
        Parameters
        ----------
        k_neighbours : int, (default = 5)
            Number of algorithms to consider during prediction.

        distance : func(xi, X) -->  np.array(), (default = None)
            User supplied distance function with args (xi, X) and returns a flat array of distances.
            If not passed (default) then the internal euclidean distance function is used.
        """
        assert k_neighbours > 0 and isinstance(k_neighbours, int), 'k_neighbours must be positive integer'
        self.k_neighbours = k_neighbours
        self.distance = distance if distance else self._euclidean_distance
        self._X_train = None
        self._y_train = None
        self.voter = None

    def fit(self, X, y):
        """Store the training feature and target data within the model for reference during prediction.

        Parameters
        ----------
        X : np.array(), shape (num_entries, num_features)
            Feature matrix for training.
            Training feature data stored here on calling `fit`.

        y : np.array(), shape (num_entries, )
            Target feature data stored here on calling `fit`.

        Returns
        -------
        None : updates {_X_train, _y_train}
        """
        self._X_train = np.array(X)
        self._y_train = np.array(y).flatten()

    @staticmethod
    def _euclidean_distance(xi, X):
        """Vectorised calculation of euclidean distances between singular coord and vector of coords.
        Supports multi-dimensional datasets.

        Parameters
        ----------
        xi : np.array(), shape(num_dimensions, )
            Singular array of coordinates for a data point.
            Each entry is the coordinate of a specific dimension.

        X : np.array(), shape(num_entries, num_dimensions)
            Array of data points to calculate the distance from.

        Returns
        -------
        distances : np.array(), shape(num_entries, )
            Array of euclidean distances between `xi` and `X`
        """
        distances = np.linalg.norm(xi - X, axis=1)
        return distances

    def _nn_data(self, xi):
        """Retrieve the data labels of the nearest `k` algorithms.
        If classification then returns the labels, regression is the continous target values.

        Parameters
        ----------
        xi : np.array(), shape(num_dimensions, )
            Singular array of coordinates for a data point.
            Each entry is the coordinate of a specific dimension.

        Returns
        -------
        labels : np.array(), shape(self.k_neighbours, )
            Data values of the nearest `k` algorithms.
        """
        distances = self._euclidean_distance(xi, self._X_train)
        neighbours = np.argsort(distances)[:self.k_neighbours]
        labels = self._y_train[neighbours]
        return labels

    def predict(self, X):
        """Predict the target value (continuous or categorical) of the passed feature data.

        Parameters
        ----------
        X : np.array(), shape(num_entries, num_dimensions)
            Array of data points to calculate the distance from.

        Returns
        -------
        y_pred : np.array(), shape(num_entries, )
            Predicted target values of the passed feature array.
        """
        y_pred = []

        for i, xi in enumerate(X):
            nn_data = self._nn_data(xi)
            prediction = self.voter(nn_data)
            y_pred.append(prediction)

        return np.array(y_pred).flatten()


class KnnClf(Knn):
    """K Nearest Neighbours Classifier.
    Inherets from  `Knn` class.

    Attributes
    ----------
    voter : function(array) --> float
        Function which takes a 1D array of integer labels and returns singular integer.
        Default is the majority categorial label, with random choice if voting tie.
    """

    def __init__(self, k_neighbours=5, distance=None, voter=None):
        """
        Parameters
        ----------
        voter : function(array) --> float
            Function which takes a 1D array of categorical labels (integers) and returns singular integer.
            Default is the majority categorial label, with random choice if voting tie.
        """
        super().__init__(k_neighbours, distance)
        self.voter = voter if voter else self._majority_vote


    @staticmethod
    def _majority_vote(neighbour_labels):
        """Returns the majority of the labels (str or int) which are algorithms.
        If there is a voting tie, a random label from the tied labels is chosen.
        `np.where` is used rather than `np.argmax` as argmax returns the lowest index by default if a tie exsits.
        This would result in an overprediction of one class versus another, hence the random choice.

        Parameters
        ----------
        neighbour_labels : np.array(), shape(num_entries, )
            The labels of the nearest algorithms, can be any type but typically expect int or str

        Returns
        -------
        labels[vote] : np.array(), shape(1, )
            Either the majority voted label or the random label of two tied labels.
        """
        labels, counts = np.unique(neighbour_labels, return_counts=True)
        majority_count = np.where(counts == counts.max())[0]  # `np.argmax` always returns the first max
        if len(majority_count) > 1:
            vote = np.random.choice(majority_count)  # random choice if more than one prediction
        else:
            vote = majority_count
        return labels[vote]


class KnnReg(Knn):
    """K Nearest Neighbours Regressor.
    Inherets from  `Knn` class.

    Attributes
    ----------
    voter : function(array) --> float
        Function which takes a 1D array of floats and returns a singular float.
        Default is `np.mean` which returns the mean of the predictions.
    """
    def __init__(self, k_neighbours=5, distance=None, voter=None):
        """
        Parameters
        ----------
        voter : function(array) --> float
            Function which takes a 1D array of floats and returns a singular float.
            Default is `np.mean` which returns the mean of the predictions.
        """
        super().__init__(k_neighbours, distance)
        self.voter = voter if voter else np.mean
