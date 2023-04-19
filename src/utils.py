import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def tune_hyperparameters(X_train, y_train, model, params):
    try:    
        grid_search = GridSearchCV(model, param_grid=params, scoring='r2', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
