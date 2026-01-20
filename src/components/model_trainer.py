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
                "RandomForestRegressor": RandomForestRegressor()
            }

            model_report = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

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
