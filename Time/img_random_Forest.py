from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

X = pd.read_csv("X_data.csv")
y = pd.read_csv("y_data.csv")
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=400)

A, b = make_regression (n_features = 2500, n_informative = 2500, random_state = 0, shuffle = True)
regr = RandomForestRegressor (n_estimators = 100, max_depth = 150, random_state = 0)
regr.fit(X_train, y_train)
RandomForestRegressor(...)
y_hat_train = regr.predict(X_train)
y_decision_train = (y_hat_train>=0.50)
y_hat_test = regr.predict(X_test)
y_decision_test = (y_hat_test>=0.50)
T = regr.feature_importances_
R_train = (y_decision_train == y_train)
Accuracy_train = np.sum(R_train) / len(y_decision_train)
R_test = (y_decision_test == y_test)
Accuracy_test = np.sum(R_test) / len(y_decision_test)
FeatImportance = regr.feature_importances_

# Output = X_test
# Output.insert(2500, "gender", y_test)
# Output.insert(2501, "gender_predicted", y_hat_test)
# Output.to_csv('RandomForestImageResults.csv')

end = time.time()
print("IMAGE-RANDOMFOREST Runtime (sec.):", end - start)
