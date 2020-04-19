import numpy as np
from sklearn.metrics import f1_score


class AnomalyDetector(object):
    """Classifier to detect anomalies in datasets based on optimal determined epsilon values.
    Assumes gaussian distribution of features.

    Attributes
    ----------
    metric : function(y_test, y_pred) --> float
        The function used to determine best epsilon value for detecting outliers.
        Must accept predicted and true values and output a score which is desired to be maximised (i.e. accuracy).

    mu : np.array(), shape(num_training_features, )
        The mean values of the passed training features.

    sigma : np.array(), shape(num_training_features, )
        The standard deviation values of the passed training features.

    epsilon : float
        Epsilon value which yields the highest metric score on the training data.

    Methods
    -------
    fit(self, X, y, ep_range) --> Determine optimal epsilon value to fit the passed training data.
    _gauss_pdf(x, mu, sigma) --> Gaussian pdf calculation to determine probability of outliers.
    predict(self, X, threshold=None) --> Predict if passed points are outliers or not..
    """

    def __init__(self, metric=f1_score):
        """
        Parameters
        ----------
        metric : function(y_test, y_pred) --> float (default = `sklearn.metrics.f1_score`)
            The function used to determine best epsilon value for detecting outliers.
            Must accept predicted and true values and output a score which is desired to be maximised (i.e. accuracy).
        """
        self.metric = metric
        self.mu, self.sigma, self.epsilon = None, None, None

    def fit(self, X, y, ep_range):
        """Determine optimal epsilon value to fit the passed training data.
        Epsilon values are selected based on metric which initialises the object.
        `mu` and `sigma` are determined as the mu and sigma of the passed feature columns.

        Parameters
        ----------
        X : np.array(), shape(num_entries, num_features)
            Feature matrix.

        y : np.array(), shape(num_entries, )
            Target labels.

        ep_range : list, shape(num_epsilon_values, )
            Epsilon values to consider for outlier determination.
            Value with highest score will be selected.

        Returns
        -------
        None
        """
        X, y = np.array(X), np.array(y)
        self.mu, self.sigma = X.mean(axis=0), X.std(axis=0)

        scores = []
        for epsilon in ep_range:
            y_pred = self.predict(X, epsilon)
            score = self.metric(y, y_pred)
            scores.append(score)

        self.epsilon = ep_range[np.argmax(scores)]

    @staticmethod
    def _gauss_pdf(x, mu, sigma):
        """Gaussian pdf calculation to determine probability of outliers.

        Parameters
        ----------
        x : np.array(), shape(num_points, )
            Data ppoints to calculate the probability of.

        mu : np.array(), shape(num_features, )
            Mean of training features.

        sigma : np.array(), shape(num_features, )
            Standard deviation of training features.

        Returns
        -------
        pdf : np.array(), shape(num_points, )
            Probabilities of the passed datapoints belonging to the gaussian distrbbutions described.
        """
        a = np.divide(1, (np.sqrt(2 * np.pi)) * sigma)
        b = np.exp(np.divide(-np.square(x - mu), 2 * np.square(sigma)))
        pdf = a * b
        return pdf

    def predict(self, X, threshold=None):
        """Predict if passed points are outliers or not.
        Allows user to specify outlier cutoff if pre-defined one is to be used.
        Optial epsilon from training will be used if one is not supplied.

        Parameters
        ----------
        X : np.array(), shape(num_entries, num_features)
            Feature matrix.

        threshold : float
            User epsilon value to use.

        Returns
        -------
        is_outlier : np.array(), shape(num_entries, )
            Boolean array indicating if a point is predicted to be an outlier or not.
        """
        epsilon = self.epsilon if not threshold else threshold
        prob = self._gauss_pdf(X, self.mu, self.sigma)
        p = np.cumprod(prob, axis=1)

        if p.ndim > 1:  # in instances where arrays of points are passed rather than individually
            p = p[:, -1]
        else:
            p = p[-1]
        is_outlier = p < epsilon
        return is_outlier
