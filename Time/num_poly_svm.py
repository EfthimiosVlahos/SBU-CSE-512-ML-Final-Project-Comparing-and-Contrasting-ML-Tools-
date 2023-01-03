
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def grid_search(param_grid_poly):
    df = pd.read_csv("/gpfs/scratch/mcirrito/CSE512_Final_Project/numerical_data.csv")
    #df = pd.read_csv("numerical_data.csv")
    y = np.array(df['gender'])
    X = df.drop('gender',axis=1)
    num_feat = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train=le.transform(y_train)

    X_test.reset_index()
    
    svc = SVC()
    clf_poly = GridSearchCV(svc, param_grid_poly)
    clf_poly.fit(X_train, y_train)
    best_score = clf_poly.best_score_
    best_params = clf_poly.best_params_
    return best_params, best_score

import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:

    params_list = [
        [{'C': [100,200,300,400,500,600,700,800,900,1000], 'gamma': [0.1], 'kernel': ['poly']},],
        [{'C': [100,200,300,400,500,600,700,800,900,1000], 'gamma': [0.05], 'kernel': ['poly']},],
        [{'C': [100,200,300,400,500,600,700,800,900,1000], 'gamma': [0.01], 'kernel': ['poly']},],
        [{'C': [100,200,300,400,500,600,700,800,900,1000], 'gamma': [0.005], 'kernel': ['poly']},],
        [{'C': [100,200,300,400,500,600,700,800,900,1000], 'gamma': [0.0001], 'kernel': ['poly']},],
        [{'C': [1100,1200,1300,1400,1500,1600,1700,1800,1900,2000], 'gamma': [0.1], 'kernel': ['poly']},],
        [{'C': [1100,1200,1300,1400,1500,1600,1700,1800,1900,2000], 'gamma': [0.05], 'kernel': ['poly']},],
        [{'C': [1100,1200,1300,1400,1500,1600,1700,1800,1900,2000], 'gamma': [0.01], 'kernel': ['poly']},],
        [{'C': [1100,1200,1300,1400,1500,1600,1700,1800,1900,2000], 'gamma': [0.005], 'kernel': ['poly']},],
        [{'C': [1100,1200,1300,1400,1500,1600,1700,1800,1900,2000], 'gamma': [0.0001], 'kernel': ['poly']},]]

    data = [[] for i in range(size)]
    i = 0
    for params in params_list:
        data[i].append(params)
        i = i + 1
        if i == size:
            i = 0
else:
    data = None

data = comm.scatter(data, root=0)

if rank >= 0:
    #print ('rank',rank,'has data:',data)
    for params in data:
        best_params, best_score = grid_search(params)
        best_of_rank = pd.DataFrame(columns = ["params", "score"])
        best_of_rank.loc[len(best_of_rank)] = [best_params, best_score]

comm.Barrier()

if rank > 0:
    comm.send(best_of_rank, dest = 0)
    
if rank == 0:
    for i in range(1, size):
        new_params = comm.recv(source = i)
        best_of_rank = pd.concat((best_of_rank, new_params),axis=0)
    print(best_of_rank)
    print("Best Parameters:")
    print(best_of_rank[best_of_rank.score == best_of_rank.score.max()])
    
