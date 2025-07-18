import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np 
import pandas as pd
import pickle
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import customException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path , exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise customException(e,sys)
    
def evaluatemodels(xtrain,ytrain,xtest,ytest,models,param):
    try:
        report={}

        for i in range(len(models)):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(xtrain,ytrain)

            model.set_params(**gs.best_params_)
            model.fit(xtrain,ytrain)  # retrains with best params

            y_pred_train=model.predict(xtrain)
            y_pred_test=model.predict(xtest)

            train_model_score = r2_score(ytrain,y_pred_train)
            test_model_score = r2_score(ytest,y_pred_test)
            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        raise customException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise customException (e,sys)
        