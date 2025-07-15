import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import customException
from src.logger import logging

from src.utils import save_object,evaluatemodels

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split Training and Test data")
            xtrain,ytrain,xtest,ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoost" : CatBoostRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "AdaBoost" : AdaBoostRegressor(),
                "Linear Regression" : LinearRegression(),
            }

            params={
                "Decision Tree" : {
                    'criterion' : ['squared_error','friedman_mse','absolute_error','poisson'] # goal to find which one is best
                },

                "Random Forest" :{
                  'n_estimators':[8,16,32,64,128,256]  # number of independent trees
                },

                "Gradient Boosting" : {
                    'learning_rate':[0.01,0.1,0.001],
                    "subsample" : [0.6,0.65,0.7,0.75,0.8,0.85],
                    "n_estimators": [8,16,32,64,128,256]  # no of sequential boosting trees
                },

                "Linear Regression" : {},

                "XGBRegressor" :{
                    "learning_rate" : [0.01,0.1,0.001],
                    "n_estimators" : [8,16,32,64,128,256]
                },

                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.001,0.1],
                    'iterations': [30, 50, 100]
                },

                "AdaBoost": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]  # number of boosting iterations
                },

                


            }

            model_report :dict=evaluatemodels(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,models=models,param = params)

            # to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise customException ("No best model found")
            
            logging.info(f"Best model found: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                
                file_path = self.model_trainer_config.trained_model_file_path,
                obj= best_model
                
            )
            predicted = best_model.predict(xtest)

            r2_square = r2_score(ytest,predicted)
            return r2_square
        
        except Exception as e:
            raise customException(e,sys)
