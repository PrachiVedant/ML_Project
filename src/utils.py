import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np 
import pandas as pd
import pickle
import dill

from sklearn.metrics import r2_score

from src.exception import customException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path , exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise customException(e,sys)
    
def evaluatemodels(xtrain,ytrain,xtest,ytest,models):
    try:
        report={}

        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(xtrain,ytrain)

            y_pred_train=model.predict(xtrain)
            y_pred_test=model.predict(xtest)

            train_model_score = r2_score(ytrain,y_pred_train)
            test_model_score = r2_score(ytest,y_pred_test)
            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        raise customException(e,sys)