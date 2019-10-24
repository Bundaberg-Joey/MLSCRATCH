import numpy as np
import matplotlib.pyplot as plt


def label_plotter_2d(X, y):
    """
    Create scatter plot of passed `X` matrix (nx2) with two colours, depending on if the associated `y` is `0` or `1`
    """
    a = X[np.where(y==0)[0]]  # filter X based on constraints
    b = X[np.where(y==1)[0]]

    xa, ya = a[:,0], a[:,1]  # variables for readability
    xb, yb = b[:,0], b[:,1]

    plt.scatter(xa, ya, color='blue', marker='o', label='Class 0', alpha=0.3)   # plot contents
    plt.scatter(xb, yb, color='orange', marker='o', label='Class 1', alpha=0.3)



def true_false_clf_plotter(X_train, y_pred, y_test):
    """
    Plots true and false predictions appropriate made by classifier with appropriate labels
    """
    comparison = y_pred == y_test
    c = X_train[np.where(comparison==1)[0]]  # filter X based on constraints
    d = X_train[np.where(comparison==0)[0]]

    xc, yc = c[:,0], c[:,1]  # variables for readability
    xd, yd = d[:,0], d[:,1]

    plt.scatter(xc, yc, color='green', marker='o', label='True', alpha=0.7)   # plot contents
    plt.scatter(xd, yd, color='red', marker='o', label='False', alpha=0.7)



def variable_sep_data(samples, shift):
    """
    Generates two different 2D gaussians with mean=0 and variance=1 and shifts one of them by `n` amount (x and y coordinates)
    """
    a = np.random.randn(samples,2)
    b = np.random.randn(samples,2)
    b += shift

    X = np.concatenate((a,b))

    y = np.zeros((X.shape[0], 1)).astype(int).flatten()  # create labels 0 for first n, 1 for second n
    y[samples:] = 1

    return X, y



def accuracy_clf(predicted, test):
    """
    calculates the similarity of two numpy arrays on a range from 0 - 1, this value is returned
    """
    return sum(predicted == test) / len(test)
