import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

def rmse(y_pred, y_test):
    """Calculaes root mean squared error"""
    mse = mean_squared_error(y_pred, y_test)
    return np.sqrt(mse)


def clean_boston_cancer():
    """
    Loads boston cancer dataset and returns pertinant features.
    """
    x, y = load_breast_cancer(return_X_y=True)
    X = x[:, :10]
    return X, y

def clean_boston_house():
    """
    Loads boston housing data
    """
    X, y = load_boston(return_X_y=True)
    return X, y

def evauluate_model(y_pred, y_test, model='clf'):
    evaluators = {
        'clf': {'accuracy': accuracy_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score},
        'reg': {'root mean squred eror': rmse}
                }

    metrics = evaluators[model]
    for m in metrics:
        func = metrics[m]
        score = func(y_pred, y_test)
        print(F'{m} = {score:.2f}')

