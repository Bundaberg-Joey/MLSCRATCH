# Needs updating to Pytest
from sklearn.model_selection import train_test_split

from algorithms.knn import KnnClf, KnnReg
from utilities import clean_boston_cancer, clean_boston_house, evauluate_model

X, y = clean_boston_cancer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
mdl = KnnClf()
mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_test)
evauluate_model(y_pred, y_test, 'clf')

X, y = clean_boston_house()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
mdl = KnnReg()
mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_test)
evauluate_model(y_pred, y_test, 'reg')
