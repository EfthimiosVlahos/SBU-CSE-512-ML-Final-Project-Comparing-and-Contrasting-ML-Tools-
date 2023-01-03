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
#from cv2 import cv2
 
df = pd.read_csv("D:/SBU/Courses/CSE512/Project/normalized_numerical_data.csv")
#df.sample(5, random_state=44)
X = df.drop(["gender"], axis=1)
y = df["gender"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=400)

# rf_model = RandomForestClassifier(n_estimators=1000, max_features="auto", random_state=500)
# rf_model.fit(X_train, y_train)
# predictions_train = rf_model.predict(X_train)
# predictions_test = rf_model.predict(X_test)
# scores_train = rf_model.score(X_train, y_train)
# scores_test = rf_model.score(X_test, y_test)
# FeatImportance = rf_model.feature_importances_

A, b = make_regression (n_features = 7, n_informative = 7, random_state = 0, shuffle = True)
regr = RandomForestRegressor (n_estimators = 350, max_depth = 50, random_state = 0)
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
# 0.0036274
# 0.0634853
# 0.0464139
# 0.29232
# 0.189607
# 0.148448
# 0.256099

# Output = X_test
# Output.insert(7, "gender", y_test)
# Output.insert(8, "gender_predicted", y_hat_test)
# Output.to_csv('RandomForestNumericalResults.csv')

auc = roc_auc_score(y_test, y_hat_test)
plt.figure()
fpr, tpr, thresholds = roc_curve(y_test, y_hat_test)
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.text(.5, .5, "AUC="+ str(auc),color='black')
plt.show()
print('AUC: %.3f' % auc)