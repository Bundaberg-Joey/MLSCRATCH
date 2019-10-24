import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from knn import knn
from MLtools import (variable_sep_data, label_plotter_2d, accuracy_clf, true_false_clf_plotter)


# load data
X, y = variable_sep_data(400, 3.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

mdl = knn()                      # initiate, fit and use model to predict labels
mdl.fit(X_train, y_train)
y_pred = mdl.predict_clf(X_test)

# plot predictions and true values out for comparison
label_plotter_2d(X_train, y_train)
true_false_clf_plotter(X_train, y_pred, y_test)
plt.legend()
plt.show()

# output model accuracy
acc = accuracy_clf(y_test, y_pred)
print(F'Model accuracy = {acc}')
