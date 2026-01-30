import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import sys

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True) # create directory if it does not exist

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj) 
    except Exception as e: 
        raise CustomException(e, sys)
    #saving preprocessor pickle in a file disk

#trains the models temporarily and return the r2 score


def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
    try:
        report = {}

        for model_name, model in models.items():
            print(f"🔍 Tuning {model_name}...")

            params = param_grids.get(model_name, {})

            if params:
                search = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    scoring="r2",
                    cv=5,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)

                best_model = search.best_estimator_
                best_params = search.best_params_
            else:
                # fallback (no hyperparams provided)
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = None

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)

            report[model_name] = {
                "r2_score": score,
                "best_params": best_params
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
