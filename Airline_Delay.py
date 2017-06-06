#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 06:40:15 2017

@author: admin
"""

# import libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import dataset
dataset = pd.read_csv('Airline.csv')
X = dataset.iloc[:,:-1].values   #: means all first is row second is column
Y = dataset.iloc[:,8].values
                
#take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy='most_frequent', axis=0) # created object of class imputer
imputer.fit(X[:,1:9]) #fit imputer in size of X
X[:,1:9] = imputer.transform(X[:,1:9])# add mean to the x matrix
imputer1 = Imputer(missing_values = 0, strategy='most_frequent', axis=0)
imputer1.fit(Y[:]) #fit imputer in size of X
Y[:] = imputer1.transform(Y[:])# add mean to the x matrix

"""#encode categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1,2,3,4])#select column to be encoded
X=onehotencoder.fit_transform(X).toarray()  #fit X to array"""


#split training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.3,random_state=0)


#fitting data in regressor object
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# prediction
y_pred = regressor.predict(X_test)