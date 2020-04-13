import numpy as np
from sklearn.metrics import f1_score


class AnomalyDetector(object):

    def __init__(self, ep_range, pdf=None):
        """
        Fits the detector to the passed data using the specified threshold values.
        `pdf` will be gaussian unless user kernel is passed (must accept (x, mu, sigma))
        """
        self.ep_range = ep_range
        self.mu, self.sigma, self.epsilon = None, None, None
        self.scores = []
        self.prob_func = self._gauss_pdf if not pdf else pdf

    def fit(self, X, y, metric=f1_score):
        """
        Determine the model parameters mu, sigma, epsilon.
        Epsilon is determined by iterating over values and selecting highest scoring.
        F1_score is default metric used but user function accepting true and predicted arrays can be passed.
        """
        X, y = np.array(X), np.array(y)
        self.mu, self.sigma = X.mean(axis=0), X.std(axis=0)

        for epsilon in self.ep_range:
            y_pred = self.predict(X, epsilon)
            self.scores.append(metric(y, y_pred))

        self.epsilon = self.ep_range[np.argmax(self.scores)]

    @staticmethod
    def _gauss_pdf(x, mu, sigma):
        """
        Calculates the gaussian PDF for the passed points
        """
        a = np.divide(1, (np.sqrt(2 * np.pi)) * sigma)
        b = np.exp(np.divide(-np.square(x - mu), 2 * np.square(sigma)))
        return a * b

    def predict(self, X, threshold=None):
        """
        Predict if passed point(s) are outliers based on threshold epsilon value.
        Model parameter used by default but user can supply own threshold.
        """
        epsilon = self.epsilon if not threshold else threshold
        prob = self.prob_func(X, self.mu, self.sigma)
        p = np.cumprod(prob, axis=1)

        if p.ndim > 1:  # in instances where arrays of points are passed rather than individually
            p = p[:, -1]
        else:
            p = p[-1]
        return p < epsilon
