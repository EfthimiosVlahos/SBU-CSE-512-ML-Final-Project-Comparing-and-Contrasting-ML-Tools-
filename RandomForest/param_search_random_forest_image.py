import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

X = pd.read_csv("X_data.csv")
y = pd.read_csv("y_data.csv")
y = np.ravel(y)
num_feat = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)

print ("Here1")

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train=le.transform(y_train)

param_grid_randomforest = [{'n_estimators': [10,20,30,40,50,60,70,80,90,100], 'max_depth': [100,150,200,250,275,300,325,350,400]},]

rf = RandomForestRegressor()
halving_cv_randomforest = HalvingGridSearchCV(
    rf, param_grid_randomforest, scoring="roc_auc", n_jobs=-1, min_resources="exhaust", factor=3
)

print ("Here2")

start = time.time()

halving_cv_randomforest.fit(X_train, y_train)
best_score_randomforest = halving_cv_randomforest.best_score_
best_params_randomforest = halving_cv_randomforest.best_params_

print("Best Parameters:", best_params_randomforest)
print("Best Score:", best_score_randomforest)
with open("param.csv", 'w') as output:
    output.write("Best Parameters : \n"+str(best_params_randomforest))
    output.write("Best Score : \n"+str(best_score_randomforest))
    output.close()

end = time.time()
print("Runtime (sec.):", end - start)

# Best Parameters: {'max_depth': 150, 'n_estimators': 100}
# Best Score: 0.9340606125944859
# Runtime (sec.): 70704.03542518616
