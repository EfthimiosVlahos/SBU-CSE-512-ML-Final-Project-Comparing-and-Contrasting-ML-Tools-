import pandas as pd
import numpy as np
#import torch.nn as nn
from sklearn.model_selection import train_test_split
from numpy import loadtxt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn import preprocessing
import math
import warnings
from tensorflow import get_logger
from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier

df = pd.read_csv("normalized_numerical_data.csv")
#df = pd.read_csv("normalized_numerical_data.csv")

print ("Here1")

df.head
X = df.drop(["gender"], axis=1)
y = np.array(df['gender'])
X.head
print(X.shape)
num_feat = X.shape[1]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=400)

print ("Here2")

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train=le.transform(y_train)
y_train
X_test.reset_index()

print ("Here3")

def generateLayersNodes(n,input_nodes, output_nodes):
    layers = []
    change = (output_nodes-input_nodes)/ (n-1)
    nodes = input_nodes
    for i in range(1,n+1):
        layers.append(math.ceil(nodes))
        nodes+=change
    return layers

def createModel(n_layers,in_nodes,out_nodes, ac, lf,lr):
    model = Sequential()
    nodes_for_layers = generateLayersNodes(n_layers,in_nodes,out_nodes)
    for i in range(1,n_layers):
        if i == 1:
            model.add(Dense(in_nodes,input_shape = (7,),activation =ac ))
        else:
            model.add(Dense(nodes_for_layers[i-1],activation =ac ))
    model.add(Dense(1,activation=ac))
    model.compile(loss=lf,optimizer=tf.keras.optimizers.Adam(learning_rate=lr),metrics=['accuracy'])
    return model

def get_clf(meta, hidden_layer_sizes, dropout):
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta["n_classes_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation="relu"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    return model

clf = KerasClassifier(
    model=get_clf,
    loss="binary_crossentropy",
    optimizer="adam",
    optimizer__lr=0.1,
    model__hidden_layer_sizes=(100,),
    model__dropout=0.5,
    verbose=False,
)

params = {
    'optimizer__lr': [0.05, 0.1],
    'model__hidden_layer_sizes': [(100, ), (50, 50, )],
    'model__dropout': [0, 0.5],
}

gs = HalvingGridSearchCV(clf, params, scoring='accuracy', n_jobs=-1, verbose=True)

print ("Here4")

gs.fit(X_train, y_train)

print ("Here5")

print(gs.best_score_, gs.best_params_)

f = open("ann_params.txt", "w")
f.write("Best Score : "+str(gs.best_score_))
f.write(str(gs.best_params_))
f.close()

print ("Here6")
