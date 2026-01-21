import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl') # path to save the trained model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() # create an instance of ModelTrainerConfig

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "XGBRegressor": XGBRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "LinearRegression": LinearRegression()
            }

            param_grids = {
                "CatBoostRegressor": {
                    "iterations": [300, 500, 800],
                    "learning_rate": [0.01, 0.05, 0.1],
                    # "depth": [4, 6, 8, 10],
                    # "l2_leaf_reg": [1, 3, 5, 7, 9],
                    # "loss_function": ["RMSE"]
                },

                "XGBRegressor": {
                    "n_estimators": [200, 500, 800],
                    "learning_rate": [0.01, 0.05, 0.1],
                    # "max_depth": [3, 5, 7, 9],
                    # "subsample": [0.6, 0.8, 1.0],
                    # "colsample_bytree": [0.6, 0.8, 1.0],
                    # "gamma": [0, 0.1, 0.3],
                    # "reg_alpha": [0, 0.1, 1],
                    # "reg_lambda": [1, 1.5, 2]
                },

                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    # "metric": ["euclidean", "manhattan", "minkowski"]
                },

                "DecisionTreeRegressor": {
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 5],
                    # "max_features": [None, "sqrt", "log2"]
                },

                "RandomForestRegressor": {
                    "n_estimators": [200, 500, 800],
                    "max_depth": [None, 10, 20, 30],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 5],
                    # "max_features": ["sqrt", "log2"],
                    # "bootstrap": [True, False]
                },
                "LinearRegression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                }

}


            model_report = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models,
                param_grids
            )
            # trains the model temporarily and gets the score
            best_model_score = max(model_report[model_name]["r2_score"] for model_name in model_report)

            best_model_name = max(model_report.keys(), key=lambda name: model_report[name]["r2_score"])

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name}")

            # 🚨 FIT THE MODEL
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
