# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:05:09 2023

@author: Bahman
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def transform_xy(X, y):
    
    
    pipe_scaler_poly = Pipeline([('std_scaler', StandardScaler()),
             ('poly_feature' , PolynomialFeatures(degree = 4))])
    
    pipe_scaler = Pipeline([('std_scaler', StandardScaler())])
    
    pip_imp = Pipeline([('miss_value' , SimpleImputer(missing_values=np.nan, strategy='mean'))])
    
    X_transform = [pip_imp, pipe_scaler_poly, pipe_scaler, pip_imp , pip_imp]
    Y_transform = [pip_imp, pip_imp, Pipeline([('std_scaler', StandardScaler())]), pip_imp, pip_imp]
    
    steps = [('miss_value' , SimpleImputer(missing_values=np.nan, strategy='mean')),
             ('std_scaler', StandardScaler()),
             ('poly_feature' , PolynomialFeatures(degree = 4))]
    y_t = []
    X_t = []
    for pip_item in X_transform:
        X_t.append(pip_item.fit_transform(X))
    
    for pip_item in Y_transform:
        y_t.append(pip_item.fit_transform(y))
    
    return X_t, y_t, X_transform, Y_transform