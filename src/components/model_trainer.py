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
                "XG Boost" : XGBRegressor(),
                "CatBoost" : CatBoostRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "KNN" : KNeighborsRegressor(),
                "AdaBoost" : AdaBoostRegressor(),
                "Linear Regression" : LinearRegression(),
            }
            model_report :dict=evaluatemodels(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,models=models)

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
