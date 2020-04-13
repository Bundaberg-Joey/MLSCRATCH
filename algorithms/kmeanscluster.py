import numpy as np


class KMeans(object):

    def __init__(self, num_clusters, max_iterations):
        self.K = num_clusters
        self.max_iters = max_iterations
        self.iters_ran = 0
        self.centroids = None
        self.losses = []
        self.prediction_labels = []

    def cluster_assignment(self, X):
        """
        Assigns each entry in X (row) to one of the passed centroid coordinates based on the euclidean distance between that
        point and the centroid.

        :param X: np.array(), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        :return: cluster_labels: np.array(int), an array of size `m`
        """
        m = X.shape[0]
        cluster_labels = np.empty(m)

        for i in range(m):
            xi = X[i]
            distances = np.linalg.norm(self.centroids - xi, axis=1)  # calc distances
            cluster_labels[i] = distances.argmin()  # get index of smallest distance
        return cluster_labels.astype(int)

    def move_centroids(self, X, cluster_labels):
        """
        Calculates the positions of the centroids based on the average of all data points belonging to the cluster.
        As potentially no data points can belong to a centroid due to initialisation, a centroid could be removed and so
        only `k-z` centroids would be in use, this alerts the user who may be expecting `k` centroids.

        :param X: np.array(), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        :param cluster_labels: np.array(int), an array of size `m`
        """
        k_labels = np.unique(cluster_labels)  # sorted np array of all cluster labels still active
        centroid_count = len(k_labels)

        if centroid_count != self.K:  # if we have k-z clusters then update number of centroids accordingly
            self.centroids = np.empty(centroid_count)
            print(F'Centroid removed, proceeding with {centroid_count} centroids')

        for i, k in enumerate(k_labels):  # find indices of points in point k
            cluster_constituents = X[cluster_labels == k]
            constituent_average = cluster_constituents.mean(axis=0)
            self.centroids[i] = constituent_average

    def loss_function(self, X, cluster_labels):
        """
        Calculates the mean squared error for each data point x_i and its centroid to monitor loss

        :param X: np.array(), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        :param cluster_labels: np.array(int), an array of size `m`
        :return: mean_squared_distance: float, the mean squared distance of all the coordinates from their centroidsKMeans.py
        """

        k_labels = np.unique(cluster_labels)
        distances = np.empty(len(k_labels))

        for i, k in enumerate(k_labels):
            xi = X[cluster_labels == k]
            centroid_location = self.centroids[i]
            k_dist = np.linalg.norm(xi - centroid_location, axis=1)
            np.concatenate((distances, k_dist))

        mean_squared_distance = np.power(distances, 2).mean()
        return mean_squared_distance

    def fit(self, X):
        """
        Divides the given feature matrix into `K` clusters based on random initialisation and subsequent refinement
        :param X: np.array(), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        """
        k_indices = np.random.choice(X.shape[0], size=self.K, replace=False)  # sample indices without replacement
        self.centroids = X[k_indices]

        while self.iters_ran < self.max_iters:
            cluster_labels = KMeans.cluster_assignment(self, X)
            KMeans.move_centroids(self, X, cluster_labels)

            iteration_loss = KMeans.loss_function(self, X, cluster_labels)
            self.losses.append(iteration_loss)
            self.iters_ran += 1
        self.prediction_labels = cluster_labels

    def predict(self, X):
        """
        Following the fitting / determination of the clusters using the `fit` method, this allows users to input new
        data points of shape `m` by `n` and predict which clusters they belong to

        :param X: np.array(), an `m` by `n` feature array with each row corresponding to a feature vector of size `n`
        :return: np.array(int), an array of size `m`
        """
        cluster_predictions = KMeans.cluster_assignment(self, X)
        return cluster_predictions
