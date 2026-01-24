import sys
from src.exception import CustomException
import pandas as pd
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl' # in data transformation we saved the preprocessor here
            print("Loading preprocessor and model...")
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            print("Preprocessor and model loaded successfully.")

            print("Transforming features...")
            data_scaled = preprocessor.transform(features)
            print("Features transformed successfully.")

            print("Making predictions...")
            preds = model.predict(data_scaled)
            print("Predictions made successfully.")
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score  

# inputs from the html form are converted to a dataframe and mapped to the values required for prediction
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {   
                "gender": self.gender,
                "race/ethnicity": self.race_ethnicity,
                "parental level of education": self.parental_level_of_education,
                "lunch": self.lunch,
                "test preparation course": self.test_preparation_course,
                "reading score": self.reading_score,
                "writing score": self.writing_score
            }
            return pd.DataFrame([custom_data_input_dict])
        except Exception as e:
            raise CustomException(e, sys)