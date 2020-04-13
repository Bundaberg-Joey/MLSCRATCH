import numpy as np
import matplotlib.pyplot as plt


class LinearRegressor(object):

    def __init__(self, add_one=True):
        self.theta = None
        self.add_one = add_one

    @staticmethod
    def _add_one(X):
        """Adds column of one to start of feature matrix (fit and predict)"""
        one = np.ones(len(X))
        X = np.c_[one, X]
        return X


    def fit(self, X, y):
        if self.add_one:
            X = self._add_one(X)

        y = y.reshape(-1, 1)
        a = np.linalg.pinv(np.matmul(X.T, X))  # pseudo inverse of the multiplcation incase non invertable
        b = np.matmul(X.T, y)
        self.theta = np.matmul(a, b)

    def predict(self, X):
        if self.add_one:
            X = self._add_one(X)
        y_pred = np.matmul(X, mdl.theta)
        return y_pred.flatten()



## rough tests
m = 100
A = np.linspace(0, 10, m)
b = (3*A + 10) + np.random.randint(0, 15, m)
C = np.linspace(0, 10, 1000)

mdl = LinearRegressor(add_one=True)
mdl.fit(A, b)
y_pred = mdl.predict(C)

plt.scatter(A, b)
plt.plot(C, y_pred)
plt.show()
