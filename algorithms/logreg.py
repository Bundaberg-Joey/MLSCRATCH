import numpy as np


class LogisticRegressor(object):

    def __init__(self, num_iters=1000, step_size=0.003):
        """
        Intialise model with default number of iterations being 1000 and a step size of 0.003
        `w` and `b` also declared for readability
        """
        self.num_iters = num_iters
        self.step_size = step_size
        self.b = np.random.randn()  # initial value can be random
        self.w = None

    @staticmethod
    def _loss(y, f):
        """
        Calculates the negative log likelyhood of a passed function `f` and its labels `y`
        """
        nll = np.sum((y * np.log(1 + np.exp(-f))) + ((1 - y) * np.log(1 + np.exp(f))))
        return nll

    @staticmethod
    def _nll_gradients(X, y, f):
        """
        Determines the gradients of the vector `w` and scalar `b` which are returned for further calculation
        """
        g = ((-y * np.exp(-f)) / (1 + np.exp(-f))) + (((1 - y) * np.exp(f)) / (1 + np.exp(f)))
        grad_w, grad_b = np.matmul(X.T, g), np.sum(g)
        return grad_w, grad_b

    def fit(self, X_train, y_train):
        """
        Determines the vector `w` and scalar `b` to fit the passed training data `X` which is a feature matrix and `y` which is a column vector of labels.
        Utilises the gradient descent method to minimise the negative logisitc loss
        """

        self.w = np.ones((X_train.shape[1], 1))  # column vector of initial values so that the transpose is a row
        losses = np.empty(self.num_iters)

        for i in range(self.num_iters):
            f = np.matmul(X_train, self.w) + self.b
            losses[i] = self._loss(y_train, f)
            grad_w, grad_b = self._nll_gradients(X_train, y_train, f)

            self.w -= (self.step_size * grad_w)  # update attributes w and b iteratively
            self.b -= (self.step_size * grad_b)

    def predict(self, X):
        """
        following fitting to the training data, returns the classification of passed column array based on the fitted `w` and `b`
        """
        f = np.matmul(X, self.w) + self.b  # determine y values based on determined w and b
        y_pred = np.empty(X.shape[0])
        y_pred[np.where(f < 0.5)[0]] = 0  # initially fill with classification 0
        y_pred[np.where(f >= 0.5)[0]] = 1  # update classification to be 1 if appropriate

        return y_pred
