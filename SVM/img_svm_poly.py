from sklearn import preprocessing
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

start = time.time()

print ("Here1")

X = pd.read_csv("X_data.csv")
y = pd.read_csv("y_data.csv")
y = np.ravel(y)

print ("Here2")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=400)
poly_svm = SVC(C=100, gamma=0.001,kernel='poly')
poly_svm.fit(X_train, y_train)
y_hat_test = poly_svm.decision_function(X_test)
y_pred = poly_svm.predict(X_test)
y_decision_test = (y_hat_test>=0)
R_test = (y_decision_test == y_test)
Accuracy_test = np.sum(R_test) / len(y_decision_test)

print ("Here3")

Output = X_test
Output.insert(2500, "gender", y_test)
Output.insert(2501, "gender_predicted", y_hat_test)
Output.to_csv('SVMPolyImageResults.csv')

print ("Here4")

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

end = time.time()
print("Runtime (sec.):", end - start)

#               precision    recall  f1-score   support

#          0.0       0.93      0.91      0.92      5806
#          1.0       0.91      0.93      0.92      5926

#     accuracy                           0.92     11732
#    macro avg       0.92      0.92      0.92     11732
# weighted avg       0.92      0.92      0.92     11732

# Runtime (sec.): 4454.703609466553


