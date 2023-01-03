from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
rbf_svm = SVC(C=60, gamma=0.01,kernel='rbf')

rbf_svm.fit(X_train, y_train)
y_hat_test = rbf_svm.decision_function(X_test)
y_pred = rbf_svm.predict(X_test)
y_decision_test = (y_hat_test>=0)
R_test = (y_decision_test == y_test)
Accuracy_test = np.sum(R_test) / len(y_decision_test)

print ("Here3")

# Output = X_test
# Output.insert(2500, "gender", y_test)
# Output.insert(2501, "gender_predicted", y_hat_test)
# Output.to_csv('SVMRbfImageResults.csv')

# print ("Here4")

# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

end = time.time()
print("IMAGE-SVM-RBF Runtime (sec.):", end - start)

#               precision    recall  f1-score   support

#          0.0       0.93      0.92      0.93      5806
#          1.0       0.93      0.93      0.93      5926

#     accuracy                           0.93     11732
#    macro avg       0.93      0.93      0.93     11732
# weighted avg       0.93      0.93      0.93     11732

# Runtime (sec.): 8881.46433711052
