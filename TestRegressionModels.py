# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 09:22:18 2023

@author: Bahman
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from myPreprocessing import transform_xy


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)


#import models
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor




# Defining Models
Model = [
    LinearRegression(),
    LinearRegression(), # For polynomial feature
    SVR(kernel = 'rbf'),
    DecisionTreeRegressor(random_state = 0),
    RandomForestRegressor(n_estimators = 10, random_state = 0)
    ]


X_t, y_t, X_transform, Y_transform = transform_xy(X, y)



# Training models on the whole dataset
for i in range(len(Model)):
    Model[i].fit(X_t[i], y_t[i])
    
    # Visualising the Model results (higher resolution)
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    pred_y=Model[i].predict(X_transform[i].transform(X_grid))
    if i ==2:
        pred_y=Y_transform[i].inverse_transform(pred_y.reshape(-1,1))
    plt.plot(X_grid, pred_y, color = 'blue')
    plt.title('Truth or Bluff (Random Forest Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

