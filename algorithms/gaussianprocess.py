import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class GPRegressor(object):

    def __init__(self, restarts=1):
        """
        Implementation of a Gaussian Process Regressor which allows for mean shift calculation.
        Conducts the hyperparameter optimisation several times and chooses the hyperparameters with the lowest nll scores.

        Sklearn implementation assumes either the mean of the process is 0 or the mean of the training sample values (bad if not properly sampled)
        This 2d implementation however allows for the mean to be shifted for calculation and then reshifted back during the prediction.
        """
        self.restarts = restarts
        self.theta = np.random.randn(4)
        self.alpha, self.sigma, self.l, self.mu = self.theta
        self.X_train = None
        self.y_train = None
        self.n = None

        assert self.restarts > 0 and isinstance(self.restarts, int), 'Restarts Must be an integer > 0'

    def _rbf_kernel(self, A, B, alpha, l):
        """
        Implementation of the the rbf kernel
        """
        n, m = len(A), len(B)
        D = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                D[i, j] = np.linalg.norm(A[i] - B[j]) ** 2

        K = (alpha ** 2) * np.exp(-D / (2 * l ** 2))
        return K

    def _nll(self, theta):
        """
        calculate the nll of the passed hyper parameters `theta` and training data
        """
        alpha, sigma, ell, mu = theta  # unpack hyperparams
        sig_theta = self._rbf_kernel(self.X_train, self.X_train, alpha, ell) + sigma ** 2 * np.identity(self.n)
        nll = 0.5 * ((self.n * np.log(2 * np.pi)) + (np.log(np.linalg.det(sig_theta))) + (
            np.dot((self.y_train - mu), np.linalg.solve(sig_theta, (self.y_train - mu)))))
        return nll

    def fit(self, X_train, y_train):
        """
        Uses `scipy.optimise.minimise` to minimise the nll to detremine ideal hyper parameters.
        If `self.restarts` > 1, run multiple optimisations and choose the hyperparameters which give the lowest score.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n = len(self.X_train)

        nll_vals = []
        params = []

        for i in range(self.restarts):
            opt = minimize(self._nll, self.theta)
            nll_vals.append(opt['fun'])
            params.append(opt['x'])

        lowest_theta = params[np.argmin(nll_vals)]
        self.alpha, self.sigma, self.l, self.mu = lowest_theta

    def _covariance_matrix(self, X):
        """
        Determine the entries of the covariance matrix
        """
        sig_aa = self._rbf_kernel(self.X_train, self.X_train, self.alpha, self.l) + self.sigma ** 2 * np.identity(
            self.n)
        sig_ab = self._rbf_kernel(self.X_train, X, self.alpha, self.l)
        sig_ba = self._rbf_kernel(X, self.X_train, self.alpha, self.l)
        sig_bb = self._rbf_kernel(X, X, self.alpha, self.l)
        return sig_aa, sig_ab, sig_ba, sig_bb

    def predict(self, X):
        """
        Predict the mean and standard deviation of the passed feature values
        """
        sig_11, sig_12, sig_21, sig_22 = self._covariance_matrix(X)

        mu = self.mu + np.matmul(sig_21, np.linalg.solve(sig_11,
                                                         self.y_train - self.mu))  # subtract from training data to recenter it and add it back to recentre the mean
        SIG = sig_22 - np.matmul(sig_21, np.linalg.solve(sig_11, sig_12))
        std = np.diag(SIG) ** 0.5
        return mu, std

    ## Tests

    # Training Data


A1 = np.hstack(([13, 12.5, 13.5], np.random.uniform(0, 10, 25), [-4]))
b1 = np.sin(2 * A1) + 0.25 * np.random.randn(len(A1))

# query data
m = 400
A2 = np.linspace(-5, 15, m)

# Fit and predict model
gpr = GPRegressor(restarts=5)
gpr.fit(A1, b1)
b2, mu_std = gpr.predict(A2)

plt.figure(figsize=(20, 5))
plt.scatter(A1, b1, color='g', marker='o', s=100, label='Training Points')
plt.plot(A2, b2, 'k-', label='Query Points')
plt.fill_between(A2, b2 - mu_std, b2 + mu_std, alpha=0.15, label='Prediction Uncertainty', color='r')
plt.xlabel('$X$', size=15)
plt.ylabel('$y$', size=15)
plt.title('Gaussian Processes Yo!', size=20)
plt.legend()
plt.show()