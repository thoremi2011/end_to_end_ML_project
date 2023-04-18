import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, tune_hyperparameters

@dataclass

class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test imput data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            param_grid = {
                'Random Forest': {
                    'model': RandomForestRegressor(),
                    'params': {
                        'n_estimators': [10, 50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'Decision Tree': {
                    'model': DecisionTreeRegressor(),
                    'params': {
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingRegressor(),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 4, 5],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                "Linear Regression": {
                    "model": LinearRegression(),
                    "params": {
                        "fit_intercept": [True, False],
                        "copy_X": [True, False],
                        "positive": [True, False],
                        "n_jobs": [-1]
                    }
                },
                'XGBRegressor': {
                    'model': XGBRegressor(),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 4, 5],
                        'min_child_weight': [1, 3, 5],
                        'gamma': [0, 0.1, 0.2]
                    }
                },
                'CatBoosting Regressor': {
                    'model': CatBoostRegressor(verbose=False),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'depth': [3, 4, 5],
                        'l2_leaf_reg': [1, 3, 5]
                    }
                },
                'AdaBoost Regressor': {
                    'model': AdaBoostRegressor(),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'loss': ['linear', 'square', 'exponential']
                    }
                }
            }

            tuned_models = {}
            for model_name, model_info in param_grid.items():
                best_model, best_score, best_params = tune_hyperparameters(X_train, y_train, model_info['model'], model_info['params'])
                tuned_models[model_name] = {
                    'model': best_model,
                    'score': best_score,
                    'params': best_params
                }
            
            best_model_name, best_model_info = max(tuned_models.items(), key=lambda x: x[1]['score'])
            best_model_score = best_model_info['score']
            best_model = best_model_info['model']

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return {
                'best_model_name': best_model_name,
                'best_model': best_model,
                'best_model_score': best_model_score, # the cross-validated R2 score from hyperparameter tuning
                'best_params': best_model_info['params'],
                'r2_square_test': r2_square, # the R2 score on the test dataset
            }

        except Exception as e:
            raise CustomException(e,sys)
