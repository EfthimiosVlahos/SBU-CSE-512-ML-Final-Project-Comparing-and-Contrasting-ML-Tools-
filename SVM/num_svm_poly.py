from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("normalized_numerical_data.csv")
X = df.drop(["gender"], axis=1)
y = df["gender"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=400)

poly_svm = SVC(C=60, gamma=0.1,kernel='poly')
poly_svm.fit(X_train, y_train)

y_hat_test = poly_svm.decision_function(X_test)
y_pred = poly_svm.predict(X_test)
y_decision_test = (y_hat_test>=0)
R_test = (y_decision_test == y_test)
Accuracy_test = np.sum(R_test) / len(y_decision_test)
#FeatImportance = poly_svm.feature_importances_
# 0.0036274
# 0.0634853
# 0.0464139
# 0.29232
# 0.189607
# 0.148448
# 0.256099

Output = X_test
Output.insert(7, "gender", y_test)
Output.insert(8, "gender_predicted", y_hat_test)
Output.to_csv('SVMPolyNumericalResults.csv')

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#               precision    recall  f1-score   support

#            0       0.97      0.99      0.98       503
#            1       0.99      0.97      0.98       498

#     accuracy                           0.98      1001
#    macro avg       0.98      0.98      0.98      1001
# weighted avg       0.98      0.98      0.98      1001
