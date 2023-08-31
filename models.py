'''
Models used in the thesis
Author: Ida Thrane (idth@itu.dk)
'''

#Import libraries
import pandas as pd
from numpy import unique, bincount
from tensorflow.random import set_seed
from sklearn.metrics import accuracy_score as m_accuracy_score
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from tensorflow.keras.losses import BinaryFocalCrossentropy
from keras import backend as K
from tensorflow import device
from tensorflow.test import gpu_device_name
device_name = gpu_device_name()
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.random import set_seed
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random
import torch

#Define geometric mean evaluation metric
def geometric_mean_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    specificity = true_negatives / (true_negatives + false_positives + K.epsilon()) #Epsilon added to ensure there is no zero-division
    sensitivity = true_positives / (true_positives + false_negatives + K.epsilon()) #Epsilon added to ensure there is no zero-division
    return K.sqrt(specificity * sensitivity) 


### Random baseline classifier
def random_model(X_train, X_test):
    predictions = []
    vals = [i[-1] for i in X_train]
    vals = list(set(vals))
    val_size = len(vals)
    
    #Set random seed
    random.seed(1)
    for i in X_test:
        index = random.randrange(val_size)
        predictions.append(vals[index])
    return predictions


### Decision tree model
def create_dt_model(max_depth, 
                    random_state, 
                    class_weight):
    #Create model
    dt_model = DecisionTreeClassifier(criterion='entropy', 
                                      max_depth=max_depth, 
                                      random_state=random_state, 
                                      class_weight = class_weight)
    
    return dt_model


###Create dense neural network
def create_nn_model(input_shape,
                    learning_rate,
                    loss=BinaryFocalCrossentropy(0, 0), 
                    model_path = None):
    
    #Clear session
    K.clear_session()
    
    #Build neural network from dense and dropout layer
    with device(device_name):
        inputs = Input(shape=(input_shape,1))
        layer = Dense(60, activation='relu')(inputs)
        layer = Dense(60, activation='relu')(inputs)
        layer = Dropout(0.3)(layer)
        layer = Dense(30, activation='relu')(layer)
        layer = Dense(30, activation='relu')(layer)
        layer = Dropout(0.3)(layer)
        layer = Flatten()(layer)
        layer = Dense(60, activation='relu')(layer)
        #Output layer
        outputs = Dense(units=1, activation='sigmoid')(layer)

        model = Model(inputs=inputs, outputs=outputs)

        #Compile model
        model.compile(loss=loss,  
                      optimizer=Adam(learning_rate=learning_rate), 
                      metrics=['accuracy', geometric_mean_metric])

        return model


###Create convolutional neural network
def create_cnn_model(input_shape,
                     learning_rate,
                     loss=BinaryFocalCrossentropy(alpha=0, gamma=0), 
                     save_model = False, 
                     model_path = None):
    
    #Clear session
    K.clear_session()
    
    #Build CNN from conolutional, maxpooling and dropout layers
    with device(device_name):
        inputs = Input(shape=(input_shape,1))
        layer = Conv1D(60, (3), activation='relu')(inputs)
        layer = Conv1D(60, (3), activation='relu')(inputs)
        layer = Dropout(0.3)(layer)
        layer = MaxPooling1D((2))(layer)
        layer = Conv1D(30, (3), activation='relu')(layer)
        layer = Conv1D(30, (3), activation='relu')(layer)
        layer = Dropout(0.3)(layer)
        layer = Flatten()(layer)
        layer = Dense(60, activation='relu')(layer)
        #Output layer
        outputs = Dense(units=1, activation='sigmoid')(layer)

        model = Model(inputs=inputs, outputs=outputs)

        #Compile model
        model.compile(loss=loss,  
                      optimizer=Adam(learning_rate=learning_rate), 
                      metrics=['accuracy', geometric_mean_metric])

        return model    