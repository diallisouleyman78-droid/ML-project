import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True) # create directory if it does not exist

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj) 
    except Exception as e: 
        raise CustomException(e, sys)
    #saving preprocessor pickle in a file disk

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            report[model_name] = score

        return report

    except Exception as e:
        raise CustomException(e, sys)
